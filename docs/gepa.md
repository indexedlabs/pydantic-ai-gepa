# GEPA Technical Specification

## Executive Summary

GEPA (Genetic Evolution with Prompt Adaptation) is an evolutionary optimization framework for text-based system components. It combines reflective mutation with genetic crossover (merge) operations to iteratively improve multi-component systems through sparse validation, Pareto front tracking, and genealogy management.

**Core Innovation**: Reflection on execution trajectories - using detailed execution traces to generate targeted improvements via LLM-based feedback.

## Validation boundary in this library

The algorithm's per-case validation scores and Pareto bookkeeping described
below belong to the harness. Reflection receives training cases, outputs,
feedback, and traces; it never receives validation cases, outputs, feedback,
side information, reports, traces, or per-case scores. Validation rollouts write
no trace file and discard captured spans before training trace collection
resumes. Aggregate validation scores and candidate adoption outcomes may be
returned to the reflector.

For managed CLI runs, the harness must load validation from outside the primary
repository and every candidate worktree, in component and git modes alike:

```bash
gepa init --agent mypkg.agents:my_agent \
  --validation-dataset /srv/gepa-heldout/my-project/validation.jsonl
```

Replace this absolute example path with the harness-owned dataset location;
never add validation data to the candidate repository. Managed runs refuse
paths inside the checkout or tracked by git, pin the external dataset identity,
and recheck it on resume and lane selection. No validation report or trace file
is written. Reflection-visible status, packets, events, final reports, and
Pareto rows contain only aggregate validation scores and outcomes. Training
reports and traces remain unredacted.

---

## 1. System Model

### 1.1 Core Abstractions

**Program/Candidate**
- A complete system instantiation: `Dict[str, str]` mapping component names to text values
- Examples: `{"system_prompt": "...", "task_instruction": "..."}`
- Components can be prompts, instructions, code snippets, or any text

**Data Types**
- `Case[InputT, OutputT, MetadataT]`: Structured dataset example from `pydantic_evals` containing inputs, optional metadata, and optional expected outputs
- `Trajectory`: Execution trace capturing intermediate system states
- `RolloutOutput`: System output (opaque to GEPA)
- `DataId`: Hashable identifier for data examples (int, str, UUID, tuple, etc.)
- `ProgramIdx`: Integer index identifying a candidate in the state

**Scoring Semantics**
- Higher scores are better
- Minibatch acceptance: Sum of scores compared
- Validation tracking: Mean of scores computed
- Scores are per-example, returned as `list[float]`

---

## 2. Core Data Structures

### 2.1 GEPAState

The persistent state tracking all optimization history.

**Required Fields**

```
program_candidates: list[dict[str, str]]
  - All discovered program variants

parent_program_for_candidate: list[list[ProgramIdx | None]]
  - Genealogy tracking: parent indices for each candidate

prog_candidate_val_subscores: list[dict[DataId, float]]
  - Sparse validation scores per candidate per validation example

pareto_front_valset: dict[DataId, float]
  - Best score achieved for each validation example

program_at_pareto_front_valset: dict[DataId, set[ProgramIdx]]
  - Set of programs achieving best score for each validation example

list_of_named_predictors: list[str]
  - Ordered list of component names

named_predictor_id_to_update_next_for_program_candidate: list[int]
  - Round-robin state tracking next component to update per candidate

i: int
  - Current iteration number (starts at -1, incremented before each iteration)

num_full_ds_evals: int
  - Number of full validation evaluations performed

total_num_evals: int
  - Total metric calls across all evaluations

num_metric_calls_by_discovery: list[int]
  - Cumulative eval count when each candidate was discovered

full_program_trace: list[dict[str, Any]]
  - Per-iteration metadata for debugging/analysis

best_outputs_valset: dict[DataId, list[tuple[ProgramIdx, RolloutOutput]]] | None
  - Optional: track best outputs per validation example

validation_schema_version: int
  - Schema version for migration support
```

**Invariants**
- All program-indexed lists must have same length
- `set(pareto_front_valset.keys()) == set(program_at_pareto_front_valset.keys())`
- All program indices in Pareto fronts < `len(program_candidates)`

**Key Methods**
- `save(run_dir)`: Persist to disk (pickle)
- `load(run_dir)`: Restore with schema migration
- `get_program_average_val_subset(idx)`: Returns (avg_score, num_samples)
- `update_state_with_new_program(...)`: Add candidate and update Pareto fronts
- `is_consistent()`: Validate invariants

### 2.2 EvaluationBatch

Container for evaluation results.

```
outputs: list[RolloutOutput]
  - System outputs per example

scores: list[float]
  - Numeric scores per example (higher is better)

trajectories: list[Trajectory] | None
  - Execution traces (only if capture_traces=True)
```

**Contract**
- `len(outputs) == len(scores) == len(batch_input)`
- If `capture_traces=True`: `len(trajectories) == len(batch_input)`

### 2.3 CandidateProposal

Proposer output structure.

```
candidate: dict[str, str]
  - The new program variant

parent_program_ids: list[int]
  - Parent program indices (1 for mutation, 2 for merge)

subsample_indices: list[DataId] | None
  - Examples used for subsample evaluation

subsample_scores_before: list[float] | None
  - Parent scores on subsample

subsample_scores_after: list[float] | None
  - New candidate scores on subsample

tag: str
  - "reflective_mutation" or "merge"

metadata: dict[str, Any]
  - Additional proposer-specific data
```

### 2.4 GEPAResult

Immutable result snapshot.

```
candidates: list[dict[str, str]]
  - All discovered programs

parents: list[list[ProgramIdx | None]]
  - Genealogy

val_aggregate_scores: list[float]
  - Average validation score per candidate

val_subscores: list[dict[DataId, float]]
  - Sparse validation scores

per_val_instance_best_candidates: dict[DataId, set[ProgramIdx]]
  - Pareto front per validation example

discovery_eval_counts: list[int]
  - Total evals when each candidate was discovered

best_outputs_valset: dict[DataId, list[tuple[ProgramIdx, RolloutOutput]]] | None
  - Optional: best outputs

total_metric_calls: int | None
num_full_val_evals: int | None
run_dir: str | None
seed: int | None
```

**Properties**
- `best_idx`: Index of highest-scoring candidate
- `best_candidate`: The program text of best
- `num_candidates`: Total discovered
- `num_val_instances`: Number of validation examples

---

## 3. Adapter Interface

The single integration point between user systems and GEPA.

### 3.1 GEPAAdapter Protocol

```python
from pydantic_evals import Case

CaseT = Case[InputT, OutputT, MetadataT]

class GEPAAdapter(Protocol[CaseT, Trajectory, RolloutOutput]):
    def evaluate(
        self,
        batch: Sequence[CaseT],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trajectory, RolloutOutput]
```

**Requirements**
- Execute candidate program on batch of examples
- Return outputs, scores, and optionally trajectories
- `len(outputs) == len(scores) == len(batch)`
- If `capture_traces=True`: return trajectories with same length
- Higher scores are better
- Never raise for individual example failures (return fallback score)
- Systemic failures (missing model, schema mismatch) can raise

```python
    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Trajectory, RolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]
```

**Requirements**
- Build JSON-serializable feedback dataset per component
- Return `dict[component_name -> list[record]]`
- Recommended record schema:
  ```
  {
    "Inputs": dict | str,           # What went into the component
    "Generated Outputs": dict | str, # What the component produced
    "Feedback": str                  # Error messages, correct answers, hints
  }
  ```

```python
    propose_new_texts: ProposalFn | None = None
```

**Requirements**
- Optional: Custom text proposal logic
- If None: GEPA uses default LLM-based proposal
- If provided: Takes candidate, reflective dataset, components → returns new texts

---

## 4. Engine Requirements

### 4.1 Main Optimization Loop

**Initialization**
1. Initialize or load state from disk
2. Evaluate seed candidate on full validation set
3. Initialize Pareto fronts with seed scores

**Main Loop** (while not stopped)
1. Increment iteration counter
2. Try merge (if scheduled and last iteration found new program):
   - Attempt merge proposal
   - Evaluate on subsample
   - Accept if `new_score >= max(parent_scores)`
   - If accepted: full validation evaluation, add to state, continue
   - If rejected but merge attempted: skip reflective mutation this iteration
3. Reflective mutation:
   - Select candidate to evolve
   - Sample training minibatch
   - Evaluate with trajectory capture
   - Select components to update
   - Build reflective dataset
   - Propose new component texts
   - Evaluate new candidate on same minibatch
   - Accept if `sum(new_scores) > sum(old_scores)`
   - If accepted: full validation evaluation, add to state, schedule merge attempts

**Full Evaluation and Addition**
- Use evaluation policy to select validation examples
- Evaluate new program
- Update state with new candidate
- Update Pareto fronts per validation example
- Compute and log aggregate score
- Identify current best program

**Stopping Conditions**
- Manual stop request
- Stop callback returns true
- Any configured stopper triggers

---

## 5. Proposer Requirements

### 5.1 ReflectiveMutationProposer

**Algorithm**

1. **Select Candidate**: Use candidate selector to choose program to evolve
2. **Sample Minibatch**: Use batch sampler to select training examples
3. **Capture Trajectories**: Evaluate current program with `capture_traces=True`
4. **Skip Conditions**:
   - No trajectories returned
   - All scores >= perfect_score (if skip_perfect_score enabled)
5. **Select Components**: Use component selector to choose what to update
6. **Build Reflective Dataset**: Call `adapter.make_reflective_dataset()`
7. **Propose New Texts**:
   - If `adapter.propose_new_texts` exists: use it
   - Else: use LLM with reflection prompt template
8. **Create New Candidate**: Copy old program, substitute new texts
9. **Evaluate New Candidate**: Evaluate on same minibatch without traces
10. **Return Proposal**: With before/after scores for acceptance test

**Configuration**
- `candidate_selector`: Which program to evolve
- `module_selector`: Which components to update ("round_robin", "all", or "reflection")
- `batch_sampler`: Training example selection
- `perfect_score`: Score threshold for skipping
- `skip_perfect_score`: Whether to skip perfect-scoring batches
- `reflection_lm`: LLM for text proposal
- `reflection_prompt_template`: Custom prompt template

### 5.2 MergeProposer

**Algorithm**

1. **Find Merge Candidates**: Identify Pareto front dominators
2. **Sample Pair**: Select two candidates i, j from dominators
3. **Find Common Ancestor**:
   - Verify neither is ancestor of the other
   - Compute intersection of ancestor sets
   - Filter ancestors:
     - Not already merged with (i, j)
     - Score <= both descendants
     - Has at least one "desirable predictor"
   - Weight-sample ancestor by score
4. **Check Desirable Predictors**: For each component:
   - Desirable if: `(ancestor == i OR ancestor == j) AND i != j`
   - Means one branch evolved, other kept ancestor
5. **Build Merged Program**: Start with ancestor, then for each component:
   - If one branch evolved: take evolved value
   - If both evolved: take from higher-scoring parent
   - If both same: keep that value
6. **Deduplicate**: Skip if merge descriptor already processed
7. **Check Validation Overlap**: Require >= N shared validation examples
8. **Select Subsample**: Stratified sampling:
   - Bucket 1: Parent 1 better
   - Bucket 2: Parent 2 better
   - Bucket 3: Tie
   - Sample evenly from buckets
9. **Evaluate**: Run merged program on subsample
10. **Return Proposal**: With parent scores and new score

**Configuration**
- `use_merge`: Enable/disable merging
- `max_merge_invocations`: Total merge attempts allowed
- `val_overlap_floor`: Minimum shared validation examples
- `merges_due`: Scheduled merge attempts (incremented on acceptance)
- `last_iter_found_new_program`: Controlled by engine

**Acceptance Criteria**
- `sum(new_scores) >= max(sum(parent1_scores), sum(parent2_scores))`

---

## 6. Component Selection Strategies

### 6.1 CandidateSelector

**Purpose**: Select which candidate to evolve in each iteration

**Interface**
```
select_candidate_idx(state: GEPAState) -> int
```

**Implementations**

**ParetoCandidateSelector**
- Remove dominated programs from Pareto fronts
- Count frequency of each program across validation instance fronts
- Sample proportionally to frequency (replicate by frequency in list)
- Programs on more fronts have higher selection probability

**CurrentBestCandidateSelector**
- Always select highest-scoring candidate
- Greedy exploitation

**EpsilonGreedyCandidateSelector**
- With probability epsilon: random selection
- With probability 1-epsilon: select best
- Balance exploration/exploitation

### 6.2 ReflectionComponentSelector

**Purpose**: Decide which component(s) to update in a candidate

**Interface**
```
__call__(
    state: GEPAState,
    trajectories: list[Trajectory],
    subsample_scores: list[float],
    candidate_idx: int,
    candidate: dict[str, str],
) -> list[str]
```

**Implementations**

**RoundRobinReflectionComponentSelector**
- Cycle through components in order
- Uses `named_predictor_id_to_update_next_for_program_candidate`
- Updates component index after selection

**AllReflectionComponentSelector**
- Return all component names
- Update everything simultaneously

### 6.3 BatchSampler

**Purpose**: Select training examples for reflective mutation

**Interface**
```
next_minibatch_ids(loader: DataLoader, state: GEPAState) -> list[DataId]
```

**Implementation**

**EpochShuffledBatchSampler**
- Shuffle all IDs at epoch boundaries using deterministic RNG
- Pad to multiple of minibatch_size with least-frequent IDs
- Return sequential slices
- Re-shuffle when epoch completes or dataset changes
- Deterministic given same seed

---

## 7. Evaluation Policy

### 7.1 EvaluationPolicy Protocol

**Purpose**: Control sparse validation scheduling and best program identification

**Interface**

```
get_eval_batch(
    loader: DataLoader,
    state: GEPAState,
    target_program_idx: ProgramIdx | None = None
) -> list[DataId]
```
- Select validation examples to evaluate for a program
- Enables sparse/adaptive validation

```
get_best_program(state: GEPAState) -> ProgramIdx
```
- Identify "best" program given sparse validation results
- May use coverage, average score, or other criteria

```
get_valset_score(program_idx: ProgramIdx, state: GEPAState) -> float
```
- Compute aggregate validation score for a program
- Used for logging and comparison

### 7.2 FullEvaluationPolicy

**Default implementation**: Evaluate all validation IDs every time

**get_eval_batch**: Returns all IDs from loader

**get_best_program**: Select by highest average score, tiebreak by coverage

**get_valset_score**: Return average of sparse scores

### 7.3 Custom Sparse Policies

Can implement:
- Adaptive sampling (focus on uncertain examples)
- Incremental evaluation (stop early if clearly worse)
- Active learning (select most informative examples)
- Budget-aware sampling (more samples for promising candidates)

---

## 8. Data Loading

### 8.1 DataLoader Protocol

**Interface**

```
all_ids() -> Sequence[DataId]
  - Return ordered universe of IDs currently available

fetch(ids: Sequence[DataId]) -> list[Case[InputT, OutputT, MetadataT]]
  - Materialize payloads for IDs, preserving order

__len__() -> int
  - Return current number of items
```

**Design Philosophy**
- DataId can be any hashable type
- Loaders can be dynamic (streaming, database-backed)
- `all_ids()` can change over time (online learning)
- Separation of identification and materialization

### 8.2 ListDataLoader

**Simple in-memory implementation**
- Wraps `Sequence[Case[InputT, OutputT, MetadataT]]`
- Uses integer indices as DataId
- Direct list access

---

## 9. Instruction Proposal

### 9.1 Default LLM-Based Proposal

**Purpose**: Generate improved component text from reflective dataset

**Default Prompt Template**

```
I provided an assistant with the following instructions to perform a task for me:
```
<curr_instructions>
```

The following are examples of different task inputs provided to the assistant
along with the assistant's response for each of them, and some feedback on how
the assistant's response could be better:
```
<inputs_outputs_feedback>
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task
description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all
niche and domain specific factual information about the task and include it in
the instruction, as a lot of it may not be available to the assistant in the
future. The assistant may have utilized a generalizable strategy to solve the
task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks.
```

**Required Placeholders**
- `<curr_instructions>`: Current component text
- `<inputs_outputs_feedback>`: Formatted reflective dataset

**Formatting Requirements**
- Convert reflective dataset to markdown
- Nested dicts/lists rendered as hierarchical headers
- Each example numbered
- Extract instruction text from code blocks in LLM response

---

## 10. Pareto Front Management

### 10.1 Per-Instance Tracking

**Concept**: Track best score and achieving programs for each validation example

**Update Logic** (per validation ID)
```
prev_score = pareto_front_valset.get(val_id, -inf)

if new_score > prev_score:
    # New leader
    pareto_front_valset[val_id] = new_score
    program_at_pareto_front_valset[val_id] = {new_program_idx}
    best_outputs_valset[val_id] = [(new_program_idx, output)]

elif new_score == prev_score:
    # Tie - add to front
    program_at_pareto_front_valset[val_id].add(new_program_idx)
    best_outputs_valset[val_id].append((new_program_idx, output))

else:
    # Worse - no change
```

### 10.2 Domination Detection

**Concept**: Program A dominates B if A is at least as good on all instances and strictly better on at least one

**Algorithm**
```
For program Y and set of other programs:
1. Find all fronts where Y appears
2. For each such front:
   - Check if any program from others is also on that front
   - If none found, Y is unique best on this instance → not dominated
3. If all fronts have a dominator from others, Y is dominated
```

### 10.3 Dominator Finding

**Purpose**: Remove redundant programs from Pareto front for selection

**Algorithm**
```
1. Count frequency of each program across instance fronts
2. Sort programs by score (ascending) - check low-scorers first
3. Iteratively:
   - For each program not yet dominated:
     - Check if dominated by remaining programs
     - If yes: mark as dominated and restart
4. Filter fronts to keep only dominators
```

### 10.4 Frequency-Weighted Sampling

**Purpose**: Select candidate proportional to breadth of excellence

**Algorithm**
```
1. Remove dominated programs
2. Count frequency per program across instance fronts
3. Build sampling list: replicate each program by its frequency
4. Random sample from list
```

**Rationale**: Programs on more fronts have higher selection probability, promoting diversity while favoring broadly capable candidates

---

## 11. State Persistence

### 11.1 Serialization Format

**Save**: Pickle dump of state dict to `gepa_state.bin`
- Include `validation_schema_version` field
- Support cloudpickle for lambda/closure serialization

**Load**: Pickle load with schema migration
- Detect version from loaded data
- Apply migrations if needed
- Validate invariants after load

### 11.2 Schema Migration

**Version 1 → Version 2**
- Convert dense validation scores (lists) to sparse (dicts)
- Convert Pareto front from list to dict keyed by DataId
- Convert best outputs from list to dict

**Requirements**
- Must preserve all candidate programs
- Must preserve genealogy
- Must preserve Pareto relationships
- Validation scores can be reconstructed from dense format

### 11.3 Resume Capability

**Initialization Logic**
```
if run_dir exists and gepa_state.bin exists:
    Load state from disk
else:
    Evaluate seed candidate on validation set
    Create fresh state with seed as candidate 0
    Initialize Pareto fronts with seed scores
    Set iteration = -1
```

**Resume Guarantees**
- Iteration counter resumes from saved value
- All candidates, scores, Pareto fronts restored
- Component round-robin state preserved
- Merge deduplication tracking maintained
- Deterministic RNG state continues sequence

---

## 12. Stopping Conditions

### 12.1 StopperProtocol

**Interface**
```
__call__(gepa_state: GEPAState) -> bool
```
Return True when optimization should stop

### 12.2 Built-in Stoppers

**MaxMetricCallsStopper**
- Stop when `state.total_num_evals >= max_metric_calls`

**MaxTrackedCandidatesStopper**
- Stop when `len(state.program_candidates) >= max_tracked_candidates`

**ScoreThresholdStopper**
- Stop when best score >= threshold

**NoImprovementStopper**
- Track best score seen
- Increment counter when no improvement
- Stop when counter >= max_iterations_without_improvement

**TimeoutStopCondition**
- Record start time
- Stop when elapsed > timeout_seconds

**FileStopper**
- Stop when file exists (e.g., `{run_dir}/gepa.stop`)

**SignalStopper**
- Register signal handlers (SIGINT, SIGTERM)
- Stop when signal received

**CompositeStopper**
- Combine multiple stoppers
- Mode "any": stop if any triggers
- Mode "all": stop if all trigger

---

## 13. Logging and Tracking

### 13.1 LoggerProtocol

**Interface**
```
log(message: str)
```

**Usage**
- Initialization messages
- Per-iteration status
- Acceptance/rejection decisions
- Best program updates
- Error messages

### 13.2 ExperimentTracker

**Interface**
```
log_metrics(metrics: dict[str, Any], step: int | None = None)
```

**Tracked Metrics**
- Base program validation score and coverage
- Per-iteration: selected candidate, subsample score, new subsample score
- Per-acceptance: new program index, validation score, coverage
- Per-merge: parent IDs, ancestor, subsample scores
- Best program score over time

**Implementation Freedom**
- Can be no-op for minimal builds
- Can integrate with any tracking system
- Not required to be wandb/mlflow specific

### 13.3 Trace Logging

**Per-iteration trace dictionary** (stored in `state.full_program_trace`)

**For reflective mutation**:
```
{
    "i": iteration,
    "selected_program_candidate": program_idx,
    "subsample_ids": list[DataId],
    "subsample_scores": list[float],
    "new_subsample_scores": list[float],
    "new_program_idx": int,
    "evaluated_val_indices": list[DataId],
}
```

**For merge**:
```
{
    "i": iteration,
    "invoked_merge": True,
    "merged": True,
    "merged_entities": (id1, id2, ancestor),
    "id1_subsample_scores": list[float],
    "id2_subsample_scores": list[float],
    "new_program_subsample_scores": list[float],
    "new_program_idx": int,
    "evaluated_val_indices": list[DataId],
}
```

---

## 14. High-Level API Requirements

### 14.1 optimize() Function

**Purpose**: Single entry point with sensible defaults

**Minimum Required Parameters**
- `seed_candidate: dict[str, str]` - Initial program
- `trainset` - Training data
- At least one stopping condition

**Optional with Defaults**
- `valset` - Defaults to trainset if not provided
- `adapter` - Created from task_lm if not provided
- `candidate_selection_strategy` - Defaults to "pareto"
- `module_selector` - Defaults to "round_robin" ("round_robin", "all", or "reflection")
- `batch_sampler` - Defaults to "epoch_shuffled"
- `reflection_minibatch_size` - Defaults to reasonable value (3-10)
- `reflection_config` - Defaults to None; when provided, configures the reflection agent (model, include_case_metadata, include_expected_output, example_bank)
- `perfect_score` - Defaults to 1.0
- `seed` - Defaults to 0
- `use_merge` - Defaults to False
- `track_component_hypotheses` - Defaults to False; when True, stores reflection hypotheses alongside each component version and feeds them back into future prompts
- `val_evaluation_policy` - Defaults to "full_eval"

**Initialization Logic**
1. Normalize datasets to DataLoader instances
2. Create adapter if not provided
3. Create stoppers from configuration
4. Create component selectors from string specs
5. Create batch sampler
6. Create proposers (reflective + optional merge)
7. Create evaluation policy
8. Create logger and experiment tracker
9. Initialize engine
10. Run optimization
11. Convert state to GEPAResult

**Return Value**: `GEPAResult` with all candidates, scores, Pareto fronts

---

## 15. Invariants and Constraints

### 15.1 State Invariants

**Structural**
- `len(program_candidates) == len(parent_program_for_candidate)`
- `len(program_candidates) == len(prog_candidate_val_subscores)`
- `len(program_candidates) == len(num_metric_calls_by_discovery)`
- `len(program_candidates) == len(named_predictor_id_to_update_next_for_program_candidate)`
- `set(pareto_front_valset.keys()) == set(program_at_pareto_front_valset.keys())`
- All program indices in Pareto fronts < `len(program_candidates)`

**Semantic**
- Iteration `i` starts at -1, incremented before each loop iteration
- `total_num_evals` is monotonically increasing
- `num_metric_calls_by_discovery[i] <= num_metric_calls_by_discovery[i+1]`

### 15.2 Evaluation Constraints

**Adapter Contract**
- `len(outputs) == len(scores) == len(batch)`
- If `capture_traces=True`: `len(trajectories) == len(batch)`
- Higher scores are better
- Never raise for individual example failures

**Scoring Semantics**
- Minibatch acceptance: `sum(scores)` comparison
- Validation tracking: `mean(scores)` over evaluated IDs
- Pareto fronts: per-instance max score tracking

### 15.3 Proposer Constraints

**Reflective Mutation**
- Requires at least one trajectory
- Skips if all scores >= perfect_score (when enabled)
- Accepts only if `sum(new_scores) > sum(old_scores)` (strict improvement)

**Merge**
- Requires at least 2 candidates
- Candidates cannot be ancestor-descendant
- Requires common ancestor with `score <= both descendants`
- Requires at least one component evolved in one branch only
- Accepts if `sum(new_scores) >= max(parent_sums)` (non-strict)

### 15.4 Pareto Front Constraints

- Each validation ID has exactly one best score
- Multiple programs can achieve that score (set membership)
- Updates are score-monotonic per instance

---

## 16. Workflow Algorithms

### 16.1 Main Loop

```
1. Initialize or load state
2. Log base program performance
3. While not stopped:
   a. Increment iteration counter
   b. Attempt merge (if scheduled):
      - If accepted: full eval + add, decrement merges_due, continue
      - If rejected: skip reflective mutation this iteration
   c. Reflective mutation:
      - Select candidate
      - Sample minibatch
      - Evaluate with traces
      - Select components
      - Build reflective dataset
      - Propose new texts
      - Evaluate new candidate
      - If improved: full eval + add, schedule merges
4. Save state
5. Return result
```

### 16.2 Reflective Mutation

```
1. Select candidate (candidate_selector)
2. Sample minibatch (batch_sampler)
3. Evaluate with capture_traces=True
4. Skip if no trajectories or all perfect
5. Select components to update (module_selector)
   - If module_selector == "reflection": the reflection agent selects components via tools (and can activate skills on demand)
6. Build reflective dataset (adapter.make_reflective_dataset)
7. Propose new texts:
   - If adapter.propose_new_texts: use it
   - Else: use LLM with reflection prompt
8. Create new candidate (substitute texts)
9. Evaluate new candidate (capture_traces=False)
10. Return proposal with scores
```

### 16.3 Merge

```
1. Find Pareto front dominators
2. Sample two candidates i, j
3. Find common ancestor:
   - Verify not ancestor-descendant
   - Compute ancestor intersection
   - Filter by score and desirability
   - Weight-sample by score
4. Build merged program:
   - Start with ancestor
   - For each component:
     * One evolved: take evolved
     * Both evolved: take from better parent
     * Both same: keep value
5. Deduplicate by descriptor
6. Check validation overlap
7. Select stratified subsample
8. Evaluate merged program
9. Return proposal
```

### 16.4 Full Evaluation and Addition

```
1. Get eval batch (evaluation_policy.get_eval_batch)
2. Evaluate program on batch
3. Increment state counters
4. Add to state (update_state_with_new_program):
   - Append candidate
   - Update Pareto fronts
   - Record parentage
   - Update component round-robin
5. Compute aggregate score
6. Identify best program
7. Log metrics
```

---

## 17. Extension Points

### 17.1 Custom Adapter

**Required**
- Implement `evaluate()` method
- Implement `make_reflective_dataset()` method

**Optional**
- Provide `propose_new_texts` for custom proposal logic

### 17.2 Custom Evaluation Policy

**Required**
- Implement `get_eval_batch()` for sparse/adaptive validation
- Implement `get_best_program()` for best selection
- Implement `get_valset_score()` for aggregate scoring

### 17.3 Custom Selectors

**CandidateSelector**: Implement `select_candidate_idx(state)`

**ReflectionComponentSelector**: Implement `__call__(state, trajectories, scores, ...)`

**BatchSampler**: Implement `next_minibatch_ids(loader, state)`

### 17.4 Custom Stopper

**Required**: Implement `__call__(state) -> bool`

Can access:
- Current iteration
- Total evaluations
- All candidates and scores
- Pareto fronts
- Full trace history

---

## 18. Performance Considerations

### 18.1 Sparse Validation Trade-offs

**Benefits**
- Faster iterations (fewer evaluations per candidate)
- More candidates explored for same budget
- Can focus on uncertain/informative examples

**Costs**
- Less accurate aggregate scores
- May need more iterations to converge
- Requires careful best-program selection logic

### 18.2 Minibatch Sizing

**Small batches (1-5)**
- Fast feedback loop
- High variance in acceptance
- More proposals tested

**Large batches (10-50)**
- Stable score estimates
- Slower iterations
- Fewer proposals tested

**Typical**: 3-10 examples for reflection

### 18.3 Merge Frequency

**More merges**
- More exploration of component combinations
- Higher computational cost
- Risk of premature convergence

**Fewer merges**
- More focused reflective improvement
- Simpler lineage tracking
- May miss beneficial combinations

**Typical**: max_merge_invocations = 5-10

### 18.4 Component Update Strategy

**Round-robin**
- Balanced improvement across components
- Simpler to reason about
- May be slower if some components don't need updates

**All**
- Faster convergence
- More LLM calls per iteration
- Higher cost per iteration

**Custom**
- Domain-specific prioritization
- Can focus on weak components
- Requires trajectory analysis

---

## 19. Implementation Checklist

### Phase 1: Core Data Structures
- [ ] GEPAState class with all fields
- [ ] State persistence (save/load)
- [ ] Schema versioning and migration
- [ ] EvaluationBatch, CandidateProposal, GEPAResult dataclasses
- [ ] Invariant validation

### Phase 2: Adapter Interface
- [ ] GEPAAdapter protocol definition
- [ ] Reference adapter implementation
- [ ] Reflective dataset format specification

### Phase 3: Selection Strategies
- [ ] CandidateSelector protocol and implementations
- [ ] ReflectionComponentSelector protocol and implementations
- [ ] BatchSampler protocol and implementation
- [ ] Deterministic RNG integration

### Phase 4: Evaluation
- [ ] DataLoader protocol and ListDataLoader
- [ ] EvaluationPolicy protocol and FullEvaluationPolicy
- [ ] Sparse validation support

### Phase 5: Proposers
- [ ] ReflectiveMutationProposer with full algorithm
- [ ] Default LLM-based text proposal
- [ ] MergeProposer with full algorithm
- [ ] Pareto front dominator detection
- [ ] Merge deduplication

### Phase 6: Pareto Management
- [ ] Per-instance front tracking
- [ ] Domination detection
- [ ] Frequency-weighted sampling

### Phase 7: Engine
- [ ] GEPAEngine initialization
- [ ] Main optimization loop
- [ ] Proposer coordination
- [ ] Full evaluation and state updates

### Phase 8: Stopping
- [ ] StopperProtocol
- [ ] All built-in stoppers
- [ ] CompositeStopper

### Phase 9: Logging
- [ ] LoggerProtocol
- [ ] ExperimentTracker interface
- [ ] Trace logging

### Phase 10: Public API
- [ ] optimize() function
- [ ] Default parameter handling
- [ ] Component factories
- [ ] Result conversion

### Phase 11: Testing
- [ ] State invariant tests
- [ ] Adapter contract tests
- [ ] Proposer algorithm tests
- [ ] Pareto front tests
- [ ] End-to-end integration tests

---

## 20. Key Design Decisions

### 20.1 Sparse Data Structures

**Decision**: Use `dict[DataId, float]` for validation scores instead of dense lists

**Rationale**:
- Supports dynamic validation sets (online learning)
- Enables sparse evaluation policies
- More efficient for large validation sets with partial coverage

### 20.2 Per-Instance Pareto Fronts

**Decision**: Track best score and programs per validation example, not global Pareto

**Rationale**:
- Programs may excel on different subsets
- Enables diversity maintenance
- Supports instance-specific debugging
- Natural for sparse validation

### 20.3 Sum-Based Acceptance

**Decision**: Accept proposals if `sum(new_scores) > sum(old_scores)` for minibatch

**Rationale**:
- Simple, interpretable criterion
- Works with variable minibatch sizes
- Strict inequality prevents churning

### 20.4 Genealogy Tracking

**Decision**: Store parent indices for all candidates

**Rationale**:
- Enables merge deduplication
- Supports lineage analysis
- Debugging and interpretability
- Minimal storage overhead

### 20.5 Iteration Counter Semantics

**Decision**: Start at -1, increment before each loop iteration

**Rationale**:
- Iteration 0 is first optimization step
- Seed evaluation happens "before" iteration 0
- Consistent with trace logging

---

## 21. Common Pitfalls and Solutions

### 21.1 Adapter Error Handling

**Pitfall**: Raising exceptions for individual example failures

**Solution**: Return fallback score (e.g., 0.0), populate trajectory with error info, never raise for per-example errors

### 21.2 Merge Infinite Loops

**Pitfall**: Repeatedly merging same programs

**Solution**: Deduplicate by merge descriptor (id1, id2, program_text_hash), track in `merges_performed`

### 21.3 Empty Trajectories

**Pitfall**: Adapter returns empty trajectories list when capture_traces=True

**Solution**: Reflective proposer returns None if no trajectories, engine continues to next iteration

### 21.4 Pareto Front Inconsistency

**Pitfall**: Program indices in fronts become invalid after state modification

**Solution**: Never mutate candidate list, only append. Validate invariants in `is_consistent()`

### 21.5 Non-Determinism

**Pitfall**: Different results with same seed

**Solution**: Use seeded RNG for all randomness (sampling, shuffling, tiebreaking), store and restore RNG state

---

## 22. Glossary

**Candidate/Program**: A complete system instantiation mapping component names to text values

**Component/Predictor**: A named text element of a program (e.g., system prompt, task instruction)

**Trajectory**: Execution trace capturing intermediate states during evaluation

**Reflective Dataset**: Structured feedback extracted from trajectories for LLM-based improvement

**Pareto Front**: Set of programs achieving best score for a validation instance

**Dominator**: Program that is on Pareto front and not dominated by other front members

**Genealogy**: Parent-child relationships between candidates

**Sparse Validation**: Evaluating candidates on subset of validation examples

**Minibatch**: Subset of training examples used for reflective mutation

**Subsample**: Subset of validation examples used for preliminary merge evaluation

**Desirable Predictor**: Component that evolved in one lineage but not the other (mergeable)

---

This specification is complete and implementation-agnostic. All requirements for recreating GEPA's functionality are documented without prescribing specific code structures.
