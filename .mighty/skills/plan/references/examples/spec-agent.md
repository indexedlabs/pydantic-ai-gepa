# Example: Workflow spec with states

```md
Summary: Document review workflow from submission through approval or rejection.

## Guarantees
- Author can submit a document for review.
- Reviewer can approve, request changes, or reject.
- Author can resubmit after addressing requested changes.
- Every state transition is recorded with actor, timestamp, and optional comment.
- Review history is append-only and never deleted.

## States
- Draft → In Review (on submit)
- In Review → Approved (on approve)
- In Review → Changes Requested (on request changes)
- In Review → Rejected (on reject)
- Changes Requested → In Review (on resubmit)
- Approved → Published (on publish)

## Constraints
- Author cannot approve their own document.
- Cannot transition from Rejected without creating a new revision.
- Cannot publish without at least one approval from a non-author.

## Rationale
- Append-only history ensures auditability for compliance requirements.
- Self-approval prohibition enforces separation of concerns.
```
