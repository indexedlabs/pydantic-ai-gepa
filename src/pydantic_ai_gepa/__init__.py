"""GEPA optimization integration for pydantic-ai."""

from __future__ import annotations

from .adapter import Adapter
from .acceptance import AcceptanceComparison, compare_candidate_samples
from .adapters.agent_adapter import (
    AgentAdapter,
    AgentAdapterTrajectory,
    SignatureAgentAdapter,
    create_adapter,
)
from .reflection import ReflectionSampler
from .cache import CacheManager, create_cached_metric
from .inspection import (
    InspectingModel,
    InspectionAborted,
    InspectionSnapshot,
)
from .exceptions import UsageBudgetExceeded
from .runner import GepaOptimizationResult, optimize_agent
from .compose import (
    FairVote,
    OmniPlan,
    PipelineResult,
    optimize_adaptive_sequential,
    optimize_best_of,
    optimize_omni,
    optimize_parallel,
    optimize_sequential,
    optimize_vote,
)
from .engines import (
    BudgetExhausted,
    BudgetTracker,
    EngineConfig,
    EngineEvent,
    EngineResult,
    OptimizationEngine,
    OptimizationTask,
    get_engine,
    list_engines,
    register_engine,
)
from .input_type import (
    BoundInputSpec,
    InputSpec,
    SignatureSuffix,
    apply_candidate_to_input_model,
    build_input_spec,
    generate_system_instructions,
    generate_user_content,
    get_gepa_components,
)
from .signature_agent import SignatureAgent
from .skills import SkillsFS
from .skills.search import (
    InMemorySkillsSearchProvider,
    LocalSkillsSearchProvider,
    SkillsSearchProvider,
)
from .types import (
    Case,
    ExampleBankConfig,
    MetadataWithMessageHistory,
    MetricResult,
    OutputT,
    ReflectionConfig,
    RolloutOutput,
    Trajectory,
)

__all__ = [
    "optimize_agent",
    "AcceptanceComparison",
    "compare_candidate_samples",
    "optimize_parallel",
    "optimize_omni",
    "optimize_adaptive_sequential",
    "optimize_best_of",
    "optimize_sequential",
    "optimize_vote",
    "PipelineResult",
    "OmniPlan",
    "FairVote",
    "GepaOptimizationResult",
    "Adapter",
    "AgentAdapter",
    "SignatureAgentAdapter",
    "ReflectionSampler",
    "CacheManager",
    "create_cached_metric",
    "Case",
    "create_adapter",
    "AgentAdapterTrajectory",
    "Trajectory",
    "RolloutOutput",
    "MetricResult",
    "MetadataWithMessageHistory",
    "ExampleBankConfig",
    "ReflectionConfig",
    "OutputT",
    "BoundInputSpec",
    "InputSpec",
    "generate_system_instructions",
    "generate_user_content",
    "get_gepa_components",
    "apply_candidate_to_input_model",
    "build_input_spec",
    "SignatureSuffix",
    "SignatureAgent",
    "SkillsFS",
    "SkillsSearchProvider",
    "LocalSkillsSearchProvider",
    "InMemorySkillsSearchProvider",
    "InspectingModel",
    "InspectionAborted",
    "InspectionSnapshot",
    "UsageBudgetExceeded",
    "OptimizationTask",
    "EngineConfig",
    "EngineResult",
    "EngineEvent",
    "BudgetTracker",
    "BudgetExhausted",
    "OptimizationEngine",
    "register_engine",
    "get_engine",
    "list_engines",
]

__version__ = "0.1.0"
