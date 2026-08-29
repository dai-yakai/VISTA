"""Case retrieval over the training set.

Implements §6 of methodology.md (case memory) and serves both:
  - the `fewshot_semantic` baseline (retrieved few-shot examples), and
  - VISTA-Net++ (case memory used as procedural prior alongside hypothesis voting).

Backbone: bge-base-en-v1.5 sentence-transformers, FAISS (cosine, normalized).
Index is keyed by question text plus task_label + complexity tag (so a
"capacity planning, level-2, remove-rank" query preferentially retrieves
similar-procedure cases, not just lexically similar ones).
"""
from __future__ import annotations

import os, json, hashlib
from typing import Sequence

import numpy as np
from loguru import logger

from . import config as C


def _index_key_for(rec: dict) -> str:
    return f"[{rec['complexity']}|{rec['op']}] {rec['question']}"


class CaseRetriever:
    def __init__(self, records: Sequence[dict], embedder, index, embeddings: np.ndarray):
        self.records = list(records)
        self.embedder = embedder
        self.index = index
        self.embeddings = embeddings

    @classmethod
    def build_or_load(cls, records: Sequence[dict], cache_dir: str | None = None) -> "CaseRetriever":
        cache_dir = cache_dir or os.path.join(C.CACHE_DIR, "retrieval")
        os.makedirs(cache_dir, exist_ok=True)
        from sentence_transformers import SentenceTransformer
        import faiss

        # Hash the inputs so we know when to invalidate cache.
        key = hashlib.md5(
            json.dumps(
                [(r["id"], r["task_label"], r["question"][:80]) for r in records],
                sort_keys=True
            ).encode()
        ).hexdigest()[:12]
        emb_file = os.path.join(cache_dir, f"emb_{key}.npy")
        idx_file = os.path.join(cache_dir, f"faiss_{key}.bin")
        meta_file = os.path.join(cache_dir, f"meta_{key}.json")

        embedder = SentenceTransformer(C.EMBED_MODEL)
        if os.path.isfile(emb_file) and os.path.isfile(idx_file):
            logger.info(f"[retrieval] loading cache {key}")
            embeddings = np.load(emb_file)
            index = faiss.read_index(idx_file)
        else:
            texts = [_index_key_for(r) for r in records]
            logger.info(f"[retrieval] embedding {len(texts)} cases with {C.EMBED_MODEL}")
            embeddings = embedder.encode(
                texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True
            ).astype("float32")
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            np.save(emb_file, embeddings)
            faiss.write_index(index, idx_file)
            with open(meta_file, "w") as f:
                json.dump({"n": len(records)}, f)
        return cls(records, embedder, index, embeddings)

    # ------------------------------------------------------------------
    def topk(
        self,
        query: str,
        k: int = C.VISTA_RETRIEVAL_K,
        *,
        tag: str | None = None,
        exclude_ids: set[int] | None = None,
    ) -> list[int]:
        """Return list of indices into `self.records` (top-k by cosine)."""
        prefix = f"[{tag}] " if tag else ""
        q_emb = self.embedder.encode(
            [prefix + query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        exclude_ids = exclude_ids or set()
        search_k = min(len(self.records), max(k, k + len(exclude_ids) + 8))
        _, idxs = self.index.search(q_emb, search_k)
        selected: list[int] = []
        for idx in idxs[0].tolist():
            if idx < 0:
                continue
            rec_id = self.records[idx].get("id")
            if rec_id in exclude_ids:
                continue
            selected.append(idx)
            if len(selected) >= k:
                break
        return selected

    def topk_records(
        self,
        query: str,
        k: int = C.VISTA_RETRIEVAL_K,
        *,
        tag: str | None = None,
        exclude_ids: set[int] | None = None,
    ) -> list[dict]:
        return [self.records[i] for i in self.topk(query, k=k, tag=tag, exclude_ids=exclude_ids)]
