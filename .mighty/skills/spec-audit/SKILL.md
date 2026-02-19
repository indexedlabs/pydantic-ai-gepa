---
name: spec-audit
description: Audit and migrate specs to the Guarantees + Constraints format. Use when asked to review spec quality, find non-conforming specs, or batch-migrate specs from the old Behaviors/Interfaces format.
---

# Spec Audit

## Overview

Systematically audit specs in the Mighty graph for format compliance, then migrate non-conforming specs to the canonical **Guarantees + Constraints** structure.

## When to use

- After adopting the new spec format and needing to migrate existing specs.
- Periodic hygiene: find specs with missing sections, stale prose, or structural issues.
- Before a major planning push: ensure the spec tree is clean and navigable.

## Audit workflow

### 1) Scan the tree

```bash
mt tree --all
```

Walk each root and its children. For every spec, run `mt show <id>` and check against the format checklist (`references/format-checklist.md`).

### 2) Identify non-conforming specs

Flag specs that have any of:

- `## Behaviors` or `## Interfaces` sections (old format)
- Missing `## Guarantees` section
- Missing `## Constraints` section (acceptable only if truly no constraints exist — annotate why)
- Unstructured prose instead of bullet lists
- Nesting deeper than 3 levels in the spec tree
- Orphan specs (no parent, not a root)

### 3) Migrate specs

Use `mt show --json <id> | jq -r '.description'` to get the current description, then transform it following the migration patterns (`references/migration-patterns.md`).

Apply with:

```bash
cat <<'EOF' | mt update <id> --description-file -
<new description>
EOF
```

### 4) Validate tree structure

After migration, verify:

- Max 3 levels deep (root → mid → leaf)
- Every non-root spec has a parent via `child_of` edge
- Root specs are area/capability buckets, not implementation details
- Parent summaries still accurately describe their children

### 5) Batch workflow

Process in tree order — roots first, then children:

1. Migrate root/parent specs first (they set the framing for children)
2. Migrate children, checking consistency with parent
3. After migrating children, re-read parent to update its summary if children changed scope
4. Leave a comment on each migrated spec: `mt comment --on <id> --content "Migrated to Guarantees + Constraints format"`

## Reference files

- Format compliance checklist: `references/format-checklist.md`
- Old → new migration patterns: `references/migration-patterns.md`
- Canonical spec template: see `.codex/skills/plan/references/spec-template.md`
- Spec examples by domain: see `.codex/skills/plan/references/examples.md`
