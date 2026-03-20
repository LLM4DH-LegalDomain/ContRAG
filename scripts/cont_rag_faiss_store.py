from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class FaissStore:
    index: faiss.Index
    dim: int
    metric: str
    meta: List[Dict[str, Any]]


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def list_jsonl_files(folder: str, exclude: Optional[List[str]] = None) -> List[str]:
    exclude = set(exclude or [])
    files = sorted(glob.glob(os.path.join(folder, "*.jsonl")))
    return [fp for fp in files if os.path.basename(fp) not in exclude]


def load_docs_from_folder(folder: str, exclude_files: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for fp in list_jsonl_files(folder, exclude=exclude_files):
        source_name = os.path.basename(fp).split(".")[0]
        for obj in iter_jsonl(fp):
            title = obj.get("title", "")
            content_list = obj.get("content", [])
            if content_list is None:
                content_list = []
            if not isinstance(content_list, list):
                content_list = [str(content_list)]
            text = "\n".join(str(x) for x in content_list)
            docs.append(
                {
                    "title": str(title) if title is not None else "",
                    "text": text,
                    "source": source_name,
                }
            )
    return docs


def load_st_model(model_path: str, device: Optional[str] = None) -> SentenceTransformer:
    return SentenceTransformer(model_path, device=device)


def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 64,
    normalize: bool = True,
    show_progress_bar: bool = False,
) -> np.ndarray:
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return np.asarray(embs, dtype=np.float32)


def build_faiss_store(
    model: SentenceTransformer,
    docs: List[Dict[str, Any]],
    batch_size: int = 64,
    normalize: bool = True,
    metric: str = "ip",
    show_progress_bar: bool = False,
) -> FaissStore:
    texts = [d["text"] for d in docs]
    X = embed_texts(
        model,
        texts,
        batch_size=batch_size,
        normalize=normalize,
        show_progress_bar=show_progress_bar,
    )
    dim = X.shape[1]

    metric_l = metric.lower()
    if metric_l == "ip":
        index = faiss.IndexFlatIP(dim)
        metric_used = "ip"
    elif metric_l == "l2":
        index = faiss.IndexFlatL2(dim)
        metric_used = "l2"
    else:
        raise ValueError("metric must be 'ip' or 'l2'")

    index.add(X)
    return FaissStore(index=index, dim=dim, metric=metric_used, meta=docs)


def save_faiss_store(store: FaissStore, out_dir: str, name: str = "faiss_store") -> str:
    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, f"{name}.index")
    meta_path = os.path.join(out_dir, f"{name}.meta.pkl")
    info_path = os.path.join(out_dir, f"{name}.info.json")

    faiss.write_index(store.index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(store.meta, f)
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump({"dim": store.dim, "metric": store.metric}, f, ensure_ascii=False, indent=2)

    return out_dir


def load_faiss_store(in_dir: str, name: str = "faiss_store") -> FaissStore:
    index_path = os.path.join(in_dir, f"{name}.index")
    meta_path = os.path.join(in_dir, f"{name}.meta.pkl")
    info_path = os.path.join(in_dir, f"{name}.info.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing FAISS index: {index_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing metadata pickle: {meta_path}")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Missing metadata JSON: {info_path}")

    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    return FaissStore(index=index, dim=int(info["dim"]), metric=str(info["metric"]), meta=meta)


def store_exists(in_dir: str, name: str = "faiss_store") -> bool:
    return all(
        os.path.exists(os.path.join(in_dir, f"{name}.{suffix}"))
        for suffix in ["index", "meta.pkl", "info.json"]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a FAISS store from legal article JSONL files, or load an existing store."
    )
    parser.add_argument("--docs-folder", type=str, help="Folder with JSONL documents.")
    parser.add_argument("--store-dir", type=str, required=True, help="Directory where the store lives.")
    parser.add_argument("--store-name", type=str, default="faiss_store", help="Base filename for the FAISS store.")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="google/embeddinggemma-300m",
        help="SentenceTransformer model path or HF id.",
    )
    parser.add_argument("--device", type=str, default=None, help="Embedding model device, e.g. cuda or cpu.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--metric", type=str, default="ip", choices=["ip", "l2"])
    parser.add_argument("--no-normalize", action="store_true", help="Disable embedding normalization.")
    parser.add_argument("--exclude-files", nargs="*", default=None, help="Optional JSONL filenames to exclude.")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild even if a store already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if store_exists(args.store_dir, args.store_name) and not args.rebuild:
        store = load_faiss_store(args.store_dir, args.store_name)
        print(json.dumps(
            {
                "status": "loaded_existing_store",
                "store_dir": args.store_dir,
                "store_name": args.store_name,
                "metric": store.metric,
                "dim": store.dim,
                "n_docs": len(store.meta),
            },
            ensure_ascii=False,
            indent=2,
        ))
        return

    if not args.docs_folder:
        raise ValueError("--docs-folder is required when building a new store.")

    docs = load_docs_from_folder(args.docs_folder, exclude_files=args.exclude_files)
    if not docs:
        raise ValueError(f"No documents found in {args.docs_folder!r}")

    model = load_st_model(args.embedding_model, device=args.device)
    store = build_faiss_store(
        model=model,
        docs=docs,
        batch_size=args.batch_size,
        normalize=not args.no_normalize,
        metric=args.metric,
        show_progress_bar=args.show_progress,
    )
    save_faiss_store(store, args.store_dir, args.store_name)

    print(json.dumps(
        {
            "status": "built_and_saved_store",
            "store_dir": args.store_dir,
            "store_name": args.store_name,
            "metric": store.metric,
            "dim": store.dim,
            "n_docs": len(store.meta),
            "embedding_model": args.embedding_model,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
