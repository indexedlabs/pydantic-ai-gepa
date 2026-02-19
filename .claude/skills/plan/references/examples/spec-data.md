# Example: Data / backend spec

```md
Summary: Chat messages persist server-side with causal ordering and immutable history.

## Guarantees
- Client can post a message to a conversation.
- Client can fetch full conversation history with cursor-based pagination.
- Client can soft-delete a message (hides from UI, retained in storage).
- Messages persist before any delivery or side effects (notifications, webhooks) occur.
- Message ordering is causal: if message A caused message B, A always appears first.
- Deleted messages are tombstoned, not removed — audit trail is preserved.
- Each message receives a stable ID at creation time that never changes.

## Constraints
- Cannot post a message to a non-existent conversation.
- Cannot mutate message content after creation (append-only).
- Maximum message size: 64KB.

## Rationale
- Persist-before-deliver prevents data loss on downstream failures.
- Append-only simplifies sync, caching, and makes audit trails reliable.
```
