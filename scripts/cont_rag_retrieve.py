from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from cont_rag_faiss_store import FaissStore, embed_texts, load_faiss_store


LABEL_RE = re.compile(r"\b(contradiction|entailment|neutral)\b", re.IGNORECASE)


def load_st_model(model_path: str, device: Optional[str] = None) -> SentenceTransformer:
    return SentenceTransformer(model_path, device=device)


def retrieve_topk(
    store: FaissStore,
    query_embs: np.ndarray,
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    if query_embs.dtype != np.float32:
        query_embs = query_embs.astype(np.float32)
    scores, idxs = store.index.search(query_embs, k)
    return scores, idxs


def load_gams3_nli_pipeline(
    adapter_dir: str,
    base_model: str = "cjvt/GaMS3-12B-Instruct",
    device_map: str = "auto",
):
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="cuda",
    )

    def build_prompt(premise: str, hypothesis: str):
        content = (
            "Si pravni asistent. Tvoja naloga je, da določiš odnos med danimi členi.\n\n"
            "Na podlagi PODANEGA BESEDILA <premise>...</premise> in TRDITVE <hypothesis>...</hypothesis> določi njun odnos.\n"
            "Izpiši NATANKO ENO oznako iz naslednjega seznama:\n\n"
            "entailment, contradiction, neutral\n\n"
            "Izpiši samo oznako, brez dodatnega besedila!\n\n"
            f"<premise>\n{premise}\n</premise>\n\n"
            f"<hypothesis>\n{hypothesis}\n</hypothesis>"
        )
        return [{"role": "user", "content": content}]

    return pipe, build_prompt


def judge_retrieved_passages(
    premise: str,
    retrieved: List[int],
    retrieved_texts: List[str],
    pline,
    prompt_fn,
) -> Dict[str, List[Tuple[int, int, str]]]:
    buckets = {
        "contradiction": [],
        "entailment": [],
        "neutral": [],
    }

    for orig_r, (doc_i, text) in enumerate(zip(retrieved, retrieved_texts), start=1):
        msg = prompt_fn(premise=premise, hypothesis=text)
        resp = pline(msg, max_new_tokens=5)
        out = resp[0]["generated_text"][-1]["content"].lower()
        m = LABEL_RE.search(out)
        label = "neutral" if m is None else m.group(1)
        buckets[label].append((doc_i, orig_r, label))

    return buckets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve top-K candidate contradictions from a FAISS store.")
    parser.add_argument("--store-dir", required=True, type=str)
    parser.add_argument("--store-name", default="faiss_store", type=str)
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="google/embeddinggemma-300m",
        help="SentenceTransformer model path or HF id.",
    )
    parser.add_argument("--device", default=None, type=str, help="Embedding model device, e.g. cuda or cpu.")
    parser.add_argument("--query", type=str, default=None, help="Single input passage/article.")
    parser.add_argument("--query-file", type=str, default=None, help="Plain-text file containing the query passage.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-normalize", action="store_true", help="Disable query embedding normalization.")
    parser.add_argument("--use-gams", action="store_true", help="Rerank top-K with the LoRA NLI model.")
    parser.add_argument("--gams-adapter-dir", type=str, default=None)
    parser.add_argument("--gams-base-model", type=str, default="cjvt/GaMS3-12B-Instruct")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to save JSON output.")
    return parser.parse_args()


def read_query(args: argparse.Namespace) -> str:
    if args.query is not None:
        return args.query
    if args.query_file is not None:
        with open(args.query_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    raise ValueError("Provide either --query or --query-file.")


def main() -> None:
    args = parse_args()
    query_text = read_query(args)

    store = load_faiss_store(args.store_dir, args.store_name)
    model = load_st_model(args.embedding_model, device=args.device)
    q_embs = embed_texts(
        model,
        [query_text],
        batch_size=args.batch_size,
        normalize=not args.no_normalize,
        show_progress_bar=False,
    )
    scores, idxs = retrieve_topk(store, q_embs, k=args.top_k)

    retrieved = idxs[0].tolist()
    base_hits: List[Dict[str, Any]] = []
    for r, doc_i in enumerate(retrieved):
        meta = store.meta[doc_i]
        base_hits.append(
            {
                "rank": r + 1,
                "score": float(scores[0, r]),
                "doc_id": int(doc_i),
                "title": meta.get("title", ""),
                "source": meta.get("source", ""),
                "content": meta.get("text", ""),
            }
        )

    output: Dict[str, Any] = {
        "query": query_text,
        "top_k": args.top_k,
        "embedding_model": args.embedding_model,
        "store_dir": args.store_dir,
        "store_name": args.store_name,
        "retrieved_docs": base_hits,
    }

    if args.use_gams:
        if not args.gams_adapter_dir:
            raise ValueError("--gams-adapter-dir is required with --use-gams")
        pline, prompt_fn = load_gams3_nli_pipeline(
            adapter_dir=args.gams_adapter_dir,
            base_model=args.gams_base_model,
        )
        retrieved_texts = [store.meta[doc_i].get("text", "") for doc_i in retrieved]
        judged = judge_retrieved_passages(query_text, retrieved, retrieved_texts, pline, prompt_fn)

        reranked: List[Dict[str, Any]] = []
        for label in ["contradiction", "entailment", "neutral"]:
            for new_rank, (doc_i, orig_r, _) in enumerate(judged[label], start=len(reranked) + 1):
                meta = store.meta[doc_i]
                reranked.append(
                    {
                        "rank": new_rank,
                        "original_rank": orig_r,
                        "label": label,
                        "score": float(scores[0, orig_r - 1]),
                        "doc_id": int(doc_i),
                        "title": meta.get("title", ""),
                        "source": meta.get("source", ""),
                        "content": meta.get("text", ""),
                    }
                )

        output["gams_reranked"] = reranked
        output["contradictions"] = [x for x in reranked if x["label"] == "contradiction"]
        output["entailment"] = [x for x in reranked if x["label"] == "entailment"]
        output["neutral"] = [x for x in reranked if x["label"] == "neutral"]

    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    print(rendered)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(rendered)


if __name__ == "__main__":
    main()
