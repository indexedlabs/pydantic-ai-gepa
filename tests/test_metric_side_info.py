from pydantic_ai_gepa.adapters.agent_adapter import AgentAdapterTrajectory
from pydantic_ai_gepa.types import RolloutOutput, MetricResult
from pydantic_ai_gepa.evaluation_models import EvaluationBatch
from pydantic_ai import Agent
from pydantic_ai_gepa.adapters.agent_adapter import create_adapter


def test_side_info_in_reflective_dataset():
    agent = Agent("test")
    adapter = create_adapter(agent=agent, metric=lambda *args: MetricResult(score=0.5))

    traj = AgentAdapterTrajectory(
        messages=[],
        final_output="test",
        metric_feedback="test feedback",
        metric_side_info={"error": "db_auth_failed", "details": {"code": 401}},
    )
    output = RolloutOutput.from_success("test")
    batch = EvaluationBatch(outputs=[output], scores=[0.5], trajectories=[traj])

    dataset = adapter.make_reflective_dataset(
        candidate={}, eval_batch=batch, components_to_update=[]
    )

    assert len(dataset.records) == 1
    assert dataset.records[0]["side_info"] == {
        "error": "db_auth_failed",
        "details": {"code": 401},
    }
