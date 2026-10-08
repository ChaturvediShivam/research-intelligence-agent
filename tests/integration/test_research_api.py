"""End-to-end M1 path: POST → background PLAN stage → persisted → GET.

`TestClient` runs BackgroundTasks synchronously on response, so by the time
POST returns the pipeline has already executed. That is what makes this a real
end-to-end assertion rather than a mock of one.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.schemas.research import ResearchPlan, RunStatus
from tests.fixtures.fake_anthropic import FakeAnthropic

QUESTION = "How large is the UK pet insurance market?"


class TestCreateRun:
    def test_accepts_and_returns_a_poll_url(
        self, client_with_fake_llm: tuple[TestClient, FakeAnthropic]
    ) -> None:
        c, _ = client_with_fake_llm
        response = c.post("/research", json={"question": QUESTION})
        assert response.status_code == 202
        body = response.json()
        assert body["run_id"].startswith("run_")
        assert body["poll_url"] == f"/research/{body['run_id']}"

    def test_rejects_a_blank_question(self, client: TestClient) -> None:
        response = client.post("/research", json={"question": "   "})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    def test_rejects_a_question_that_cannot_be_decomposed(self, client: TestClient) -> None:
        response = client.post("/research", json={"question": "aaaaaaaaaaaaaa"})
        assert response.status_code == 422

    def test_rejects_out_of_range_max_sources(self, client: TestClient) -> None:
        response = client.post("/research", json={"question": QUESTION, "max_sources": 500})
        assert response.status_code == 422

    def test_missing_body_is_rejected(self, client: TestClient) -> None:
        assert client.post("/research", json={}).status_code == 422


class TestPollRun:
    def test_unknown_run_is_404_with_the_error_envelope(self, client: TestClient) -> None:
        response = client.get("/research/run_doesnotexist")
        assert response.status_code == 404
        body = response.json()
        assert body["error"]["code"] == "not_found"
        assert body["error"]["details"]["run_id"] == "run_doesnotexist"


class TestPlanningOverHttp:
    """Planning, persistence and cost over HTTP.

    `client_with_fake_llm` scripts the planning call only, so these runs fail
    at discovery. That is the point: the plan a failed run produced must still
    be persisted and returned. A completed run is covered in
    test_research_api_pipeline.py, against the full fake pipeline.
    """

    def test_plan_is_produced_persisted_and_returned(
        self, client_with_fake_llm: tuple[TestClient, FakeAnthropic]
    ) -> None:
        c, fake = client_with_fake_llm
        run_id = c.post("/research", json={"question": QUESTION}).json()["run_id"]

        detail = c.get(f"/research/{run_id}").json()

        # The run reaches discovery and fails there, because this fake cannot
        # search. Before M10 the route stopped after planning and reported
        # COMPLETED, so a plan-only run was indistinguishable from a finished
        # one (F-017).
        assert detail["status"] == RunStatus.FAILED.value
        assert detail["question"] == QUESTION
        assert detail["error"].startswith("discover:")

        # The plan survived a round trip through SQLite and still validates,
        # even though the run as a whole failed after producing it.
        plan = ResearchPlan.model_validate(detail["plan"])
        assert len(plan.sub_questions) == 2
        assert [sq.id for sq in plan.ordered()] == ["SQ1", "SQ2"]
        assert plan.assumptions == ["'UK' includes Northern Ireland."]

        # Exactly one model call for stage 1.
        assert len(fake.messages.calls) == 1

    def test_measured_cost_is_reported_not_estimated(
        self, client_with_fake_llm: tuple[TestClient, FakeAnthropic]
    ) -> None:
        c, _ = client_with_fake_llm
        run_id = c.post("/research", json={"question": QUESTION}).json()["run_id"]

        cost = c.get(f"/research/{run_id}").json()["cost"]
        assert cost is not None
        assert cost["total_usd"] > 0
        # Attribution per stage is what makes the routing decision reviewable.
        # `plan` spent; `assess` is the free stage that always runs, so a
        # failed run still reports its gaps. Discovery failed before spending
        # anything. A completed run attributes cost across every stage —
        # asserted in test_research_api_pipeline.py.
        assert set(cost["by_stage"]) == {"plan", "assess"}
        assert cost["by_stage"]["assess"] == 0.0
        assert cost["input_tokens"] == 1000
        assert cost["output_tokens"] == 500
        assert cost["total_duration_ms"] >= 0
        # No cache on a first call, so the hit rate is a real 0.0, not None.
        assert cost["cache_hit_rate"] == 0.0

    def test_two_runs_are_independent(
        self, client_with_fake_llm: tuple[TestClient, FakeAnthropic]
    ) -> None:
        c, fake = client_with_fake_llm
        first = c.post("/research", json={"question": QUESTION}).json()["run_id"]
        # The fake has one scripted plan, so the second run fails a stage
        # earlier than the first. The two runs must not share state.
        second = c.post("/research", json={"question": QUESTION}).json()["run_id"]

        assert first != second
        first_detail = c.get(f"/research/{first}").json()
        second_detail = c.get(f"/research/{second}").json()

        # Both fail, but at different stages — which is the independence this
        # test is for: run 1 got the scripted plan, run 2 got nothing.
        assert first_detail["plan"] is not None
        assert first_detail["error"].startswith("discover:")
        assert second_detail["plan"] is None
        assert second_detail["error"].startswith("plan:")
        del fake


class TestFailurePath:
    def test_transport_failure_marks_the_run_failed_and_names_the_stage(
        self, client_with_failing_llm: TestClient
    ) -> None:
        """A background task that died silently would leave a run stuck forever."""
        c = client_with_failing_llm
        run_id = c.post("/research", json={"question": QUESTION}).json()["run_id"]

        detail = c.get(f"/research/{run_id}").json()
        assert detail["status"] == RunStatus.FAILED.value
        assert detail["plan"] is None
        assert detail["error"] is not None
        assert detail["error"].startswith("plan:")

    def test_failure_detail_does_not_leak_internals(
        self, client_with_failing_llm: TestClient
    ) -> None:
        c = client_with_failing_llm
        run_id = c.post("/research", json={"question": QUESTION}).json()["run_id"]
        text = c.get(f"/research/{run_id}").text
        assert "Traceback" not in text


class TestOpenAPI:
    def test_both_research_routes_are_documented(self, client: TestClient) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        assert "/research" in paths
        assert "/research/{run_id}" in paths
