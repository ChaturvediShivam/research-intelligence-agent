"""Embeddings.

`Embedder` is the seam ADR-003 promised: local ONNX today, a hosted embedder
later if measurement shows retrieval is the binding constraint. The protocol
exists because that swap is a real possibility with a decision criterion
attached, not because abstraction is tidy.

Two properties the rest of the system relies on:

- **Determinism.** The same text must embed to the same vector, or an index
  built in one process disagrees with a query embedded in another, and eval
  results stop being comparable between runs.
- **Normalised vectors.** Embeddings are L2-normalised, so cosine similarity
  and inner product coincide and `sqlite-vec`'s distance can be reasoned
  about directly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import structlog

logger = structlog.get_logger(__name__)

# BAAI/bge-small-en-v1.5. Hardcoded rather than discovered at runtime because
# the vector table's column width is fixed at schema creation: a model change
# is a reindex, and it should be a deliberate, visible edit.
DEFAULT_DIMENSION = 384

# bge models are trained with an asymmetric prefix: queries are embedded with
# this instruction, documents without it. Omitting it measurably degrades
# retrieval, and it is the kind of detail that silently costs recall.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors.

    `embed_documents` and `embed_query` are separate because asymmetric models
    require different treatment of each, and collapsing them into one method
    makes that impossible to express.
    """

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def l2_normalise(vector: list[float]) -> list[float]:
    """Scale a vector to unit length. A zero vector is returned unchanged."""
    magnitude = sum(component * component for component in vector) ** 0.5
    if magnitude == 0.0:
        return vector
    return [component / magnitude for component in vector]


class FastEmbedEmbedder:
    """Local ONNX embeddings via fastembed (ADR-003).

    The model is loaded lazily and once: construction is cheap, so the class
    can be instantiated in a factory without paying for a model load that may
    never be used.
    """

    def __init__(
        self, model_name: str = "BAAI/bge-small-en-v1.5", *, dimension: int = DEFAULT_DIMENSION
    ) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._model: object | None = None

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_model(self) -> object:
        if self._model is None:
            from fastembed import TextEmbedding

            logger.info("embedding_model_load", model=self._model_name)
            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for indexing. No query prefix."""
        if not texts:
            return []
        model = self._get_model()
        vectors = [
            l2_normalise([float(x) for x in vector])
            for vector in model.embed(texts)  # type: ignore[attr-defined]
        ]
        if vectors and len(vectors[0]) != self._dimension:
            # A silent dimension mismatch would corrupt the vector table.
            raise ValueError(
                f"{self._model_name} produced {len(vectors[0])}-dimensional "
                f"vectors; the index expects {self._dimension}. Changing the "
                "model requires a reindex and a schema change."
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a query, with the asymmetric instruction prefix applied."""
        prefixed = f"{BGE_QUERY_PREFIX}{text}" if "bge" in self._model_name.lower() else text
        return self.embed_documents([prefixed])[0]
