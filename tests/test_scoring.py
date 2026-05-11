from mega_ai.core.context import SharedContext
from mega_ai.evals.scoring import score_case


def test_score_baseline() -> None:
    ctx = SharedContext(job_id="j1", user_query="What is the capital of France?")
    report = score_case(final_answer="Paris is the capital.", context=ctx, expected="Paris", category="baseline")
    assert report.dimensions[0].score >= 0.5
