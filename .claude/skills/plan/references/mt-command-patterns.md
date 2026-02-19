# mt command patterns (multiline-safe)

## Create a spec with rich markdown description (stdin)

```bash
cat <<'EOF' | mt new --title "Spec title" --type feature --description-file -
Summary: <what this capability IS, present tense, 1–2 sentences>

## Guarantees
- <everything the system promises — capabilities, behaviors, visual treatments, data properties>

## Constraints
- <everything the system prevents — prohibitions, limitations, error conditions>

## Non-goals
- <explicitly out of scope>

## Related (optional)
- See [related spec](cite:mt-spec-...) / [related decision](cite:mt-dec-...) as needed.
EOF
```

## Create a child spec (then link hierarchy via edges)

```bash
cat <<'EOF' | mt new --title "Child spec title" --type feature --description-file -
Summary: ...
EOF

mt link --from <child-spec-id> --rel child_of --to-spec <parent-spec-id>
```

## Create a task linked to a spec/decision (spawned edge)

```bash
cat <<'EOF' | mt task new --title "Implement X" --type task --source <spec-or-decision-id> --description-file -
Summary: ...

## Acceptance Criteria
- [ ] ...
EOF
```

## Record a decision (ADR-style fields)

```bash
mt decision new \
  --title "Decision title" \
  --context "What problem are we solving?" \
  --decision "What did we decide?" \
  --rationale "Why this choice?" \
  --alternatives "What else did we consider?" \
  --consequences "Trade-offs and implications"
```

## Add a progress comment

```bash
mt comment --on <spec-or-task-id> --content "Found X; choosing Y because Z. Next: do A then B."
```

## Link evidence (files/commits)

```bash
mt link --from <spec-or-task-id> --rel implemented_by --to-type file --to-ref path/to/file.py -d "Core implementation"
mt link --from <spec-or-task-id> --rel implemented_by --to-type commit --to-ref <sha> -d "Primary change"
```
