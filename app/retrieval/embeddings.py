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

# How many chunks are embedded in one ONNX forward pass.
#
# This is a memory bound, not a throughput tuning knob. Transformer attention
# allocates activations proportional to batch x sequence^2, and stage 4 embeds
# every chunk of every source in a single `embed_documents` call. fastembed's
# own default batch size is 256, which for 512-token chunks asks ONNX for
# several gigabytes at once: on Render's 512 MB instance the kernel
# SIGKILLed the container mid-stage, killing the background task with no
# Python exception to catch and no log line (F-019).
#
# Measured in a 512 MB container, embedding 512-token chunks. The first
# block isolates this function; the second drives the real `run_retrieve_stage`
# (which also builds the sqlite-vec index and retrieves), and is the number
# that matters for the deployed service:
#
#   chunks   batch   peak RSS   what
#      250     256   OOM-kill   this function (also OOM at a 1 GB ceiling)
#      250      16   OOM-kill   this function
#      250       8    409 MB    this function
#      250       4    322 MB    this function
#     1000       4    317 MB    this function
#      250       4    341 MB    full stage 4
#     1000       4    351 MB    full stage 4
#
# At a fixed batch size peak memory is flat in chunk count, which is the
# property that matters: chunk count is unbounded (it follows source length),
# so memory must not scale with it. 4 is chosen over 8 for the headroom —
# the difference in wall time was 13.1 s against 12.7 s for 250 chunks, which
# is noise, while the difference in peak is 87 MB on a 512 MB budget.
DEFAULT_BATCH_SIZE = 4


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
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        dimension: int = DEFAULT_DIMENSION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self._model_name = model_name
        self._dimension = dimension
        self._batch_size = batch_size
        self._model: object | None = None

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def batch_size(self) -> int:
        return self._batch_size

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
        # batch_size is passed explicitly: fastembed's default is 256, which
        # is what OOM-killed the container (see DEFAULT_BATCH_SIZE).
        vectors = [
            l2_normalise([float(x) for x in vector])
            for vector in model.embed(texts, batch_size=self._batch_size)  # type: ignore[attr-defined]
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
