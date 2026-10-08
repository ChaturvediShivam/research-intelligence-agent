"""Error envelope shape and leak-prevention on unexpected exceptions."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.errors import (
    AppError,
    CostCeilingExceededError,
    NotFoundError,
    PipelineStageError,
    UnsafeURLError,
    register_exception_handlers,
)


@pytest.fixture
def error_app() -> FastAPI:
    """A throwaway app whose routes raise each error type on demand."""
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/not-found")
    async def _nf() -> None:
        raise NotFoundError("Run 'abc' does not exist.", details={"run_id": "abc"})

    @app.get("/unsafe")
    async def _unsafe() -> None:
        raise UnsafeURLError("URL resolves to a private address.")

    @app.get("/ceiling")
    async def _ceiling() -> None:
        raise CostCeilingExceededError("Run would exceed the cost ceiling.")

    @app.get("/stage")
    async def _stage() -> None:
        raise PipelineStageError("retrieve", "Vector store unavailable.")

    @app.get("/boom")
    async def _boom() -> None:
        raise RuntimeError("DB password is hunter2 and the stack is internal")

    return app


class TestStatusMapping:
    @pytest.mark.parametrize(
        ("path", "status", "code"),
        [
            ("/not-found", 404, "not_found"),
            ("/unsafe", 400, "unsafe_url"),
            ("/ceiling", 402, "cost_ceiling_exceeded"),
            ("/stage", 500, "pipeline_stage_error"),
        ],
    )
    def test_each_error_maps_to_its_status_and_code(
        self, error_app: FastAPI, path: str, status: int, code: str
    ) -> None:
        with TestClient(error_app, raise_server_exceptions=False) as c:
            response = c.get(path)
        assert response.status_code == status
        assert response.json()["error"]["code"] == code

    def test_details_are_returned_when_present(self, error_app: FastAPI) -> None:
        with TestClient(error_app, raise_server_exceptions=False) as c:
            body = c.get("/not-found").json()
        assert body["error"]["details"] == {"run_id": "abc"}

    def test_pipeline_error_names_the_stage(self, error_app: FastAPI) -> None:
        with TestClient(error_app, raise_server_exceptions=False) as c:
            body = c.get("/stage").json()
        assert body["error"]["details"]["stage"] == "retrieve"


class TestUnexpectedExceptionIsOpaque:
    def test_internal_detail_never_reaches_the_client(self, error_app: FastAPI) -> None:
        """An unhandled error must not leak internals in the response body."""
        with TestClient(error_app, raise_server_exceptions=False) as c:
            response = c.get("/boom")
        assert response.status_code == 500
        body = response.json()
        assert body["error"]["code"] == "internal_error"
        assert body["error"]["message"] == "An internal error occurred."
        assert "hunter2" not in response.text
        assert "RuntimeError" not in response.text
        assert "Traceback" not in response.text


class TestHierarchy:
    def test_every_error_is_an_apperror(self) -> None:
        for cls in (NotFoundError, UnsafeURLError, CostCeilingExceededError):
            assert issubclass(cls, AppError)

    def test_validation_error_422_on_bad_body(self) -> None:
        """Pydantic failures are normalised into the same envelope."""
        app = FastAPI()
        register_exception_handlers(app)

        from pydantic import BaseModel

        class Body(BaseModel):
            n: int

        @app.post("/echo")
        async def _echo(body: Body) -> dict[str, int]:
            return {"n": body.n}

        with TestClient(app, raise_server_exceptions=False) as c:
            response = c.post("/echo", json={"n": "not-a-number"})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"
