from __future__ import annotations

import math
import re
from collections import Counter

from .anonymize import anonymize_text
from .schema import CharacterSpec, Comment, Probe, RetrievedComment
from .vector_store import ExactVectorStore, isolate_text_transformers_runtime


def tokenize(text: str) -> list[str]:
    lowered = text.casefold()
    latin = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", lowered)
    cjk_runs = re.findall(r"[\u3400-\u9fff]+", lowered)
    cjk = []
    for run in cjk_runs:
        cjk.extend(run)
        cjk.extend(run[index : index + 2] for index in range(len(run) - 1))
    return latin + cjk


def bm25_scores(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    if not documents:
        return []
    tokenized = [tokenize(document) for document in documents]
    query_tokens = list(dict.fromkeys(tokenize(query)))
    lengths = [len(tokens) for tokens in tokenized]
    avgdl = sum(lengths) / max(1, len(lengths))
    frequencies = [Counter(tokens) for tokens in tokenized]
    document_frequency = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))
    scores = []
    n_docs = len(documents)
    for index, freq in enumerate(frequencies):
        score = 0.0
        for term in query_tokens:
            tf = freq.get(term, 0)
            if not tf:
                continue
            df = document_frequency[term]
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denominator = tf + k1 * (1 - b + b * lengths[index] / max(avgdl, 1e-9))
            score += idf * tf * (k1 + 1) / denominator
        scores.append(score)
    return scores


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, document_index in enumerate(ranking, 1):
            fused[document_index] = fused.get(document_index, 0.0) + 1.0 / (k + rank)
    return fused


class DenseRetriever:
    def __init__(self, vector_store: str, embedding_model_key: str) -> None:
        self.vector_store = ExactVectorStore(
            vector_store,
            expected_model_key=embedding_model_key,
        )

    def scores(
        self,
        *,
        probe: Probe,
        comments: list[Comment],
        spec: CharacterSpec,
    ) -> list[float]:
        return self.vector_store.scores(
            probe=probe,
            language=spec.language,
            comments=comments,
        )


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        try:
            isolate_text_transformers_runtime()
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "reranking needs the optional 'dense' dependencies: pip install -e '.[dense]'"
            ) from exc
        self.model = CrossEncoder(model_name)

    def scores(self, query: str, documents: list[str]) -> list[float]:
        values = self.model.predict([(query, document) for document in documents])
        return [float(value) for value in values]


class HybridRetriever:
    def __init__(
        self,
        *,
        mode: str = "auto",
        vector_store: str | None = None,
        embedding_model_key: str = "qwen3-embedding-0.6b",
        reranker_model: str | None = None,
        bm25_top_k: int = 20,
        dense_top_k: int = 20,
        final_top_k: int = 10,
    ) -> None:
        if mode not in {"auto", "bm25", "hybrid"}:
            raise ValueError("retrieval mode must be auto, bm25, or hybrid")
        self.mode = mode
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k
        self.final_top_k = final_top_k
        self.dense: DenseRetriever | None = None
        self.reranker: CrossEncoderReranker | None = None
        if mode in {"auto", "hybrid"}:
            try:
                if not vector_store:
                    raise RuntimeError("hybrid retrieval requires a frozen vector_store")
                self.dense = DenseRetriever(vector_store, embedding_model_key)
                if reranker_model:
                    self.reranker = CrossEncoderReranker(reranker_model)
            except RuntimeError:
                if mode == "hybrid":
                    raise

    @property
    def effective_mode(self) -> str:
        if self.dense and self.reranker:
            return "bm25+qwen3-vector+rrf+rerank"
        return "bm25+qwen3-vector+rrf" if self.dense else "bm25"

    def retrieve(
        self,
        probe: Probe,
        comments: list[Comment],
        spec: CharacterSpec,
    ) -> list[RetrievedComment]:
        documents = [comment.raw_text for comment in comments]
        query = probe.query(spec.language)
        scores = bm25_scores(query, documents)
        bm25_ranking = sorted(range(len(comments)), key=lambda i: (-scores[i], comments[i].comment_id))[
            : self.bm25_top_k
        ]
        rankings = [bm25_ranking]
        source_ranks: dict[int, list[str]] = {index: ["bm25"] for index in bm25_ranking}
        if self.dense:
            dense_scores = self.dense.scores(
                probe=probe,
                comments=comments,
                spec=spec,
            )
            dense_ranking = sorted(
                range(len(comments)), key=lambda i: (-dense_scores[i], comments[i].comment_id)
            )[: self.dense_top_k]
            rankings.append(dense_ranking)
            for index in dense_ranking:
                source_ranks.setdefault(index, []).append("dense")
        fused = reciprocal_rank_fusion(rankings)
        candidates = sorted(fused, key=lambda i: (-fused[i], comments[i].comment_id))
        final_scores = dict(fused)
        if self.reranker:
            rerank_scores = self.reranker.scores(
                query, [documents[index] for index in candidates]
            )
            final_scores = {
                index: rerank_scores[position] for position, index in enumerate(candidates)
            }
            candidates = sorted(
                candidates,
                key=lambda i: (-final_scores[i], comments[i].comment_id),
            )
            for index in candidates:
                source_ranks.setdefault(index, []).append("reranker")
        final = candidates[: self.final_top_k]
        return [
            RetrievedComment(
                comment_id=comments[index].comment_id,
                anonymous_id=spec.anonymous_id,
                text=anonymize_text(comments[index].raw_text, spec),
                platform=comments[index].platform,
                author_hash=comments[index].author_hash,
                language=comments[index].language,
                score=final_scores[index],
                rank_sources=source_ranks[index],
            )
            for index in final
        ]
