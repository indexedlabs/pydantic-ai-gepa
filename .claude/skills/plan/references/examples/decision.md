# Example: Decision

Use this when you're choosing between alternatives or setting constraints.

- **Title**: "Use server-side sessions instead of JWTs for auth"
- **Context**: "We need authentication for the web app. Immediate session revocation is a hard security requirement. JWTs are stateless but revocation requires workarounds. Server-side sessions need a store but give us immediate control."
- **Decision**: "Use server-side sessions stored in Redis with a 24-hour TTL. Session ID is sent as an HttpOnly cookie."
- **Rationale**: "Immediate revocation is non-negotiable for our threat model. Redis gives sub-millisecond lookups with well-understood ops patterns. HttpOnly cookies prevent XSS token theft."
- **Alternatives**:
  - "JWTs with short expiry + refresh tokens (rejected: revocation still has a window, refresh flow adds client complexity)"
  - "JWTs with a server-side blocklist (rejected: reintroduces statefulness without the simplicity of sessions)"
- **Consequences** (contract impact):
  - "Adds guarantee: any session can be revoked and takes effect on the next request"
  - "Adds constraint: Redis is a runtime dependency for all authenticated requests"
  - "Adds constraint: auth tokens are HttpOnly cookies, not accessible to client-side JavaScript"
  - "Trades off: cannot deploy to a fully stateless edge without session replication"
