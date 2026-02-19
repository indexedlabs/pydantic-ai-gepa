# Example: Task (execution)

```md
Summary: Add session-based authentication middleware to the API server.

## Context
- Source spec: [User authentication](cite:xx-spec-...)
- Related decision(s): [Server-side sessions over JWTs](cite:xx-dec-...)

## Goal
- Protect API routes with session validation without breaking existing public endpoints.

## Acceptance Criteria
- [ ] Guarantee: public routes (health check, login, signup) are accessible without a session
- [ ] Guarantee: authenticated routes return 401 when no valid session cookie is present
- [ ] Guarantee: session revocation takes effect on the very next request
- [ ] Constraint: session is validated against the store on every request (no local caching)
- [ ] Constraint: session cookie is HttpOnly (not accessible to client-side JS)
- [ ] Includes integration tests covering: valid session, expired session, revoked session, missing cookie, public route bypass

## Out of Scope
- Session revocation UI (separate task).
- OAuth / SSO integration.
```
