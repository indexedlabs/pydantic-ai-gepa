# mt command patterns (task workflow)

## Create a task with rich markdown description (stdin)

```bash
cat <<'EOF' | mt task new --title "Task title" --type task --source <spec-or-decision-id> --description-file -
Summary: <1–2 sentences>

## Goal
- <what we’re building/fixing>

## Acceptance Criteria
- [ ] <concrete check>

## Notes
- <approach, key files, risks>
EOF
```

## Claim, comment, close

```bash
mt claim <task-id> --reason "Starting work"
mt comment <task-id> --content "Checkpoint: ..."
mt task close <task-id> --reason "Done" --resolution "What changed + how to verify"
```

## Link evidence (files/commits)

```bash
mt link --from <task-or-spec-id> --rel implemented_by --to-type file --to-ref path/to/file.py -d "Core implementation"
mt link --from <task-or-spec-id> --rel tested_by --to-type file --to-ref api/tests/test_foo.py -d "Coverage"
mt link --from <task-or-spec-id> --rel implemented_by --to-type commit --to-ref <sha> -d "Primary change"
```
