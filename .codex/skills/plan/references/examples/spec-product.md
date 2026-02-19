# Example: High-level product spec (top of a large tree)

```md
Summary: A collaborative project tracker where teams organize work into boards, lists, and cards.

## Guarantees
- User can create boards and invite team members.
- User can organize cards into lists and drag to reorder or move between lists.
- User can assign members, set due dates, and add labels to cards.
- User can comment on cards with rich text and file attachments.
- Every card belongs to exactly one list; every list belongs to exactly one board.
- Card ordering within a list is stable — reordering one card does not shuffle others.
- All mutations are attributed to a user and timestamped.
- Board state is eventually consistent across all connected clients within 2 seconds.

## Constraints
- A board cannot have more than 500 lists (prevents degenerate usage).
- Deleted boards are soft-deleted and recoverable for 30 days.
- Guest users can view but not modify board content.

## Non-goals
- Gantt charts or timeline views (separate capability).
- Real-time collaborative editing within card descriptions (v1 uses last-write-wins).
```
