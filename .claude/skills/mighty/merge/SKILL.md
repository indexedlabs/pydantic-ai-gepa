---
name: mighty-merge
description: "Resolve `.mighty/loro/*.loro` merge conflicts safely using `mt merge`, then verify store health and sync."
---

# Mighty: Resolve Loro Merge Conflicts

Use this when git reports conflicts in `.mighty/loro/*.loro` files.

## Workflow

1. Confirm what’s conflicted:
   - `git status`
   - `git diff --name-only --diff-filter=U`

2. Use Mighty’s merge helper (preferred):
   - `mt merge` (merges all unresolved `.mighty/loro/*.loro` conflicts)
   - Or `mt merge .mighty/loro/edges/<file>.loro ...` (merge specific files)

3. Re-check health:
   - `mt doctor`

4. Commit and sync:
   - `mt commit` (commits `.mighty/` changes)
   - `git push`

## Notes

- Avoid resolving `.loro` conflicts by hand; they are binary CRDT snapshots. Use `mt merge`.
- If `mt merge` can’t resolve a conflict, keep the work unblocked: capture details in an issue comment and mark the blocking item appropriately.
