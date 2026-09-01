from __future__ import annotations

from typing import Protocol

from .anonymize import anonymize_text
from .connected_retrieval import CohereReranker, RerankerSettings
from .schema import CharacterSpec, Comment, Probe, RetrievedComment
from .vector_store import ExactVectorStore


class Retriever(Protocol):
    @property
    def effective_mode(self) -> str: ...

    def retrieve(
        self,
        probe: Probe,
        comments: list[Comment],
        spec: CharacterSpec,
    ) -> list[RetrievedComment]: ...


class VectorRerankRetriever:
    """Exact-vector candidate recall followed by connected Cohere reranking."""

    def __init__(
        self,
        *,
        vector_store: str,
        embedding_model_key: str,
        reranker: CohereReranker,
        candidate_top_k: int = 20,
        final_top_k: int = 10,
    ) -> None:
        if candidate_top_k <= 0 or final_top_k <= 0:
            raise ValueError("retrieval Top-K values must be positive")
        if final_top_k > candidate_top_k:
            raise ValueError("final_top_k cannot exceed candidate_top_k")
        self.vector_store = ExactVectorStore(
            vector_store,
            expected_model_key=embedding_model_key,
        )
        self.embedding_model_key = embedding_model_key
        self.reranker = reranker
        self.candidate_top_k = candidate_top_k
        self.final_top_k = final_top_k

    @property
    def effective_mode(self) -> str:
        return "openai_exact_vector_recall+cohere_rerank"

    def retrieve(
        self,
        probe: Probe,
        comments: list[Comment],
        spec: CharacterSpec,
    ) -> list[RetrievedComment]:
        if not comments:
            return []
        query = probe.query(spec.language)
        dense_scores = self.vector_store.scores(
            probe=probe,
            comments=comments,
            language=spec.language,
        )
        candidates = sorted(
            range(len(comments)),
            key=lambda index: (-dense_scores[index], comments[index].comment_id),
        )[: self.candidate_top_k]
        candidate_documents = [
            anonymize_text(comments[index].raw_text, spec) for index in candidates
        ]
        rerank_scores = self.reranker.scores(
            query,
            candidate_documents,
        )
        final_scores = {
            index: rerank_scores[position]
            for position, index in enumerate(candidates)
        }
        candidates.sort(
            key=lambda index: (-final_scores[index], comments[index].comment_id)
        )
        return [
            RetrievedComment(
                comment_id=comments[index].comment_id,
                anonymous_id=spec.anonymous_id,
                text=anonymize_text(comments[index].raw_text, spec),
                platform=comments[index].platform,
                author_hash=comments[index].author_hash,
                language=comments[index].language,
                score=final_scores[index],
                rank_sources=["exact_vector", "cohere_rerank"],
            )
            for index in candidates[: self.final_top_k]
        ]


class DeterministicSmokeRetriever:
    """Network-free deterministic fixture for non-research unit and smoke runs."""

    def __init__(self, final_top_k: int = 10) -> None:
        if final_top_k <= 0:
            raise ValueError("final_top_k must be positive")
        self.final_top_k = final_top_k

    @property
    def effective_mode(self) -> str:
        return "deterministic_smoke_only"

    def retrieve(
        self,
        probe: Probe,
        comments: list[Comment],
        spec: CharacterSpec,
    ) -> list[RetrievedComment]:
        del probe
        ordered = sorted(comments, key=lambda comment: comment.comment_id)[
            : self.final_top_k
        ]
        return [
            RetrievedComment(
                comment_id=comment.comment_id,
                anonymous_id=spec.anonymous_id,
                text=anonymize_text(comment.raw_text, spec),
                platform=comment.platform,
                author_hash=comment.author_hash,
                language=comment.language,
                score=1.0 / rank,
                rank_sources=["deterministic_smoke_only"],
            )
            for rank, comment in enumerate(ordered, 1)
        ]


def build_retriever(
    *,
    mode: str,
    vector_store: str | None,
    embedding_model_key: str,
    reranker_model: str,
    env_file: str,
    rerank_trace_path: str | None = None,
    candidate_top_k: int = 20,
    final_top_k: int = 10,
) -> Retriever:
    if mode == "deterministic_smoke":
        return DeterministicSmokeRetriever(final_top_k=final_top_k)
    if mode != "vector_rerank":
        raise RuntimeError(
            "research retrieval mode must be vector_rerank; "
            "deterministic_smoke is allowed only for non-research tests"
        )
    if not vector_store:
        raise RuntimeError("vector_rerank requires a frozen vector store")
    settings = RerankerSettings.from_env(env_file)
    if settings.model != reranker_model:
        raise RuntimeError(
            f"configured reranker {settings.model!r} differs from {reranker_model!r}"
        )
    reranker = CohereReranker(settings, trace_path=rerank_trace_path)
    return VectorRerankRetriever(
        vector_store=vector_store,
        embedding_model_key=embedding_model_key,
        reranker=reranker,
        candidate_top_k=candidate_top_k,
        final_top_k=final_top_k,
    )
