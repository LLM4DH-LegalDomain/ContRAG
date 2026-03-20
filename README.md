# ContRAG: Contradiction-Aware Retrieval for Legal Texts

This repository contains a minimal, reproducible implementation of the **ContRAG** pipeline for contradiction-aware retrieval in legal corpora.

The framework combines:
- Dense retrieval over legal articles using FAISS
- Optional NLI-based reranking using a domain-adapted model (GaMS)
- Article-level structuring of legal documents

The repository is designed to be lightweight and modular, allowing users to:
1. Build or load a FAISS index over legal texts
2. Retrieve top-k relevant articles for a given query
3. Optionally rerank retrieved candidates using a contradiction-aware NLI model

---

## Repository Structure

```
.
├── scripts/
│   ├── cont_rag_faiss_store.py
│   └── cont_rag_retrieve.py
│
├── data/
│   ├── SLawNLI.jsonl
│   └── demo_split/
│       ├── KZ-1.jsonl
│       ├── SPZ.jsonl
│       ├── ZVOP-2.jsonl
│       └── OZ.jsonl
│
├── README.md
└── requirements.txt
```

---

## Data Format

The framework operates at the article level.

Each input file must be a `.jsonl` where each line represents a single legal article:

```
{
  "id": "SPZ_123",
  "title": "Člen 123",
  "content": [
    "Prvi odstavek...",
    "Drugi odstavek..."
  ]
}
```

During indexing:
- `title` and `content` are concatenated into a single text field
- Each entry becomes one retrieval unit

Important:
- The system assumes article-level granularity
- If using external datasets, they must be preprocessed into this format

---

## Installation

```
pip install -r requirements.txt
```

---

## Building a FAISS Store

To build an index from a dataset:

```
python scripts/cont_rag_faiss_store.py   --docs-folder data/demo_split   --store-dir data/store   --store-name demo_store   --embedding-model models/sentence_transformer   --device cuda   --show-progress
```

To load an existing store:

```
python scripts/cont_rag_faiss_store.py   --store-dir data/store   --store-name demo_store
```

---

## Retrieval

Basic retrieval:

```
python scripts/cont_rag_retrieve.py   --store-dir data/store   --store-name demo_store   --embedding-model models/sentence_transformer   --device cuda   --query "Besedilo člena..."   --top-k 10
```

---

## Retrieval with GaMS Reranking

```
python scripts/cont_rag_retrieve.py   --store-dir data/store   --store-name demo_store   --embedding-model models/sentence_transformer   --device cuda   --query "Besedilo člena..."   --top-k 20   --use-gams   --gams-adapter-dir models/gams_adapter
```

Pipeline:
1. Retrieve top-k candidates using dense retrieval
2. Apply NLI classification between query and retrieved articles
3. Group results into CONTRADICTION, ENTAILMENT, and NEUTRAL

---

## Output

The retrieval script outputs JSON:

```
{
  "query": "...",
  "results": [
    {
      "id": "...",
      "score": 0.83,
      "label": "CONTRADICTION",
      "text": "..."
    }
  ]
}
```

Optional:
```
--json-out results.json
```

---

## Demo Data

The `data/demo_split` folder contains four Slovenian laws split into articles:
- KZ-1
- SPZ
- ZVOP-2
- OZ

These can be used to build a FAISS index and run retrieval experiments.

The `SLawNLI.jsonl` dataset contains annotated NLI pairs and can be used for evaluation, fine-tuning, or analysis.

---

## Notes

- The system is article-centric and assumes legal texts are split into articles
- Retrieval is semantic (dense)
- Reranking is optional and enables contradiction-aware reasoning
- The framework is modular and supports alternative models and extensions

---

## Acknowledgements

This work was conducted as part of the projects Large Language Models for Digital Humanities (LLM4DH, GC-0002) and Basic Research for the Development of Spoken Language Resources and Speech Technologies for the Slovenian Language (MEZZANINE, J7-4642), financed from the national budget by a contract between the Slovenian Research Agency and the Faculty of Computer and Information Science, University of Ljubljana.
