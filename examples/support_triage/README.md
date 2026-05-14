# Support triage — `pydantic-ai-gepa` external-reflection demo

A small support agent with 5 intentionally vague tools (`open_ticket`,
`lookup_order`, `escalate_to_human`, `send_reset_link`, `check_shipment_status`).
The agent routes a customer message to one of them; the dataset says which
tool should have been called. As tool descriptions improve, routing accuracy
goes up.

Run it end-to-end via the `gepa` CLI (added by [pydanticaigepa-spec-973](cite:pydanticaigepa-spec-973)).
**You** are the reflection model — the CLI scores the baseline and writes a
per-case report; you read the report and edit slots or source code.

## Setup

```bash
# Default model is openai:gpt-4o-mini — put OPENAI_API_KEY in .env (gepa
# auto-loads it). Override the example's model via GEPA_EXAMPLE_MODEL, e.g.
# 'openai:gpt-5-mini' or 'groq:llama-3.3-70b-versatile'.

cd <repo root>

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

The default substring metric won't score routing correctly. We have a metric
that checks whether the agent routed to the expected tool — register it as a
**top-level** `metric =` line in `.gepa/gepa.toml` (NOT inside `[defaults]`):

```toml
agent = "examples.support_triage.agent:agent"
dataset = "examples/support_triage/dataset.jsonl"
metric = "examples.support_triage.metric:routed_to_expected_tool"

[defaults]
# minibatch_size = 10
```

## Run the loop

```bash
# Score the current baseline. The vague tool descriptions cause routing
# failures — expect mean_score well below 1.0.
gepa eval --size 6 --seed 0 --max-iterations 50

# The summary line at the bottom of stdout points at the report:
#   "report_path": ".gepa/runs/<run_id>/reports/<candidate_id>.md"
cat .gepa/runs/<run_id>/reports/<candidate_id>.md

# Rewrite the slot that's hurting the most.
cat > /tmp/lookup_desc.md <<'EOF'
Look up an order by its order id (formats: A-NNNN, B-NN, C-NN, etc.).
Use this when the customer references a specific order number and asks
about it (status, ETA, contents, etc.).
EOF
gepa components set tool:lookup_order:description --content-file /tmp/lookup_desc.md

# Re-eval the new baseline against the same minibatch for a clean A/B.
# The minibatch_id is in the previous summary line.
gepa eval --minibatch-id <id> --run-id <run_id>

# When you've got a baseline you like, commit it.
git add .gepa/components/
git commit -m "tune support-triage tool descriptions"

# Inspect the run history.
gepa pareto --format tsv --all
```

## Add a tool mid-session

Edit `agent.py` to add a 6th `@agent.tool` (e.g. `cancel_subscription`). Then:

```bash
gepa eval
# -> exits 2: "Found unconfirmed component slots; refusing to evaluate the baseline."
# -> writes stubs at .gepa/staged/tool:cancel_subscription:description.md, etc.

# Review and confirm (or override the docstring seed):
echo "Cancel an active subscription by user email. Use when the customer wants to stop billing." > /tmp/cs.md
gepa components confirm tool:cancel_subscription:description --content-file /tmp/cs.md
gepa components confirm tool:cancel_subscription:param:email

gepa eval   # now works
```

## What this exercises

| `gepa` surface | This demo uses it |
|---|---|
| `gepa init` | Scaffolds `.gepa/` + seeds 11 component slots |
| `gepa components list/show/set` | Inspecting + editing tool descriptions |
| `gepa eval` | Scores the baseline + writes per-case failure report |
| `gepa eval --candidate-file` | A/B-tests an explicit candidate JSON without writing through to `.gepa/components/` |
| `gepa pareto --format tsv` | Tracking which iterations helped |
| `metric` ref in `gepa.toml` | Custom routing-accuracy scorer |
| Stage-and-confirm | Adding a 6th `@agent.tool` makes `gepa eval` refuse until you confirm the new slots |
| `.env` auto-load | `OPENAI_API_KEY` is picked up automatically |
