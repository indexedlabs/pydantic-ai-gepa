# Support triage — `pydantic-ai-gepa` external-reflection demo

A small support agent with 5 intentionally vague tools (`open_ticket`,
`lookup_order`, `escalate_to_human`, `send_reset_link`, `check_shipment_status`).
The agent routes a customer message to one of them; the dataset says which
tool should have been called. As tool descriptions improve, routing accuracy
goes up.

Run it end-to-end via the `gepa` CLI (added by [pydanticaigepa-spec-973](cite:pydanticaigepa-spec-973)).

## Setup

```bash
# Default model is openai:gpt-4o-mini — set OPENAI_API_KEY in .env first.
# (You can override via GEPA_EXAMPLE_MODEL, e.g. 'openai:gpt-5-mini' or
# 'groq:llama-3.3-70b-versatile'.)

cd <repo root>

# Point the CLI at the example agent + dataset.
gepa init --agent examples.support_triage.agent:agent --dataset examples/support_triage/dataset.jsonl --force
```

`gepa init` introspects the agent and pre-seeds `.gepa/components/` with the
existing instruction text + each tool's docstring + each parameter description.

```bash
gepa components list
# Lists 11 slots:
#   instructions
#   tool:open_ticket:description
#   tool:open_ticket:param:summary
#   tool:lookup_order:description
#   tool:lookup_order:param:order_id
#   tool:escalate_to_human:description
#   tool:escalate_to_human:param:reason
#   tool:send_reset_link:description
#   tool:send_reset_link:param:email
#   tool:check_shipment_status:description
#   tool:check_shipment_status:param:tracking_number
```

## Wire up the custom metric

The default substring metric (`gepa eval` with no custom metric) won't score
this correctly. We have a metric that checks whether the agent routed to the
expected tool — register it via `.gepa/gepa.toml`:

```bash
cat >> .gepa/gepa.toml <<'EOF'
metric = "examples.support_triage.metric:routed_to_expected_tool"
EOF
```

## Run the loop

```bash
# Baseline eval — expect mid-range accuracy because the tool descriptions
# are vague.
gepa propose --minibatch-size 6 --seed 0 --max-iterations 10

# Read the report; pick a slot that failed most often and rewrite it.
cat .gepa/runs/<run_id>/proposals/<proposal_id>.report.md

# Edit one of the tool descriptions to something better.
cat > /tmp/lookup_desc.md <<'EOF'
Look up an order by its order id (formats: A-NNNN, B-NN, C-NN, etc.).
Use this when the customer references a specific order number and asks
about it (status, ETA, contents, etc.).
EOF
gepa components set tool:lookup_order:description --content-file /tmp/lookup_desc.md

# Re-eval the candidate (the proposal file is already on disk from `gepa propose`).
gepa eval --candidate-file .gepa/runs/<run_id>/proposals/<proposal_id>.json

# If it improves, adopt as the new baseline.
gepa eval --candidate-file ... | grep mean_score    # compare
gepa apply --candidate-file .gepa/runs/<run_id>/proposals/<proposal_id>.json --commit

# Look at the Pareto history.
gepa pareto --format tsv --all
```

## Try in-process LLM reflection

Want the library to *propose* the edits instead of editing slots by hand?
Pass `--reflection-model`:

```bash
gepa propose --minibatch-size 6 --seed 0 --reflection-model openai:gpt-4o
```

The proposal candidate at `.gepa/runs/<run_id>/proposals/<id>.json` will now
contain LLM-rewritten slot text. Eval, apply, repeat.

## What this exercises

| `gepa` surface | This demo uses it |
|---|---|
| `gepa init` | Scaffolds .gepa/ + seeds 11 component slots |
| `gepa components list/show/set` | Inspecting + editing tool descriptions |
| `gepa propose` | Baseline eval + emits per-case failure report |
| `gepa eval --candidate-file` | Re-scoring edited candidates |
| `gepa apply --commit` | Promoting a winning candidate to baseline |
| `gepa pareto --format tsv` | Tracking which iterations helped |
| `metric` ref in `gepa.toml` | Custom routing-accuracy scorer |
| Stage-and-confirm | Add a 6th `@agent.tool` to `agent.py` and re-run `gepa propose` — it will refuse and write a stub for the new slot |
