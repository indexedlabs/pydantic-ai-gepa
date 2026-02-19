# Example: API spec tree (progressive detail via tree depth)

### Top-level (PM level)

```md
Summary: Per-organization API rate limits with user-facing errors and admin visibility.

## Guarantees
- Caller can check remaining rate limit quota via response headers.
- Org admin can view current usage and historical limit events.
- Every API request is classified into exactly one rate-limit bucket.
- Rate limit state is never shared across organizations.

## Constraints
- Exceeding the limit returns HTTP 429 (never silently drops or queues).

## Rationale
- Central rate limiting prevents per-service divergence and gives consistent error shapes.
- See [Enforce rate limits at gateway](cite:xx-dec-...) for full ADR.

## Non-goals
- Per-endpoint custom thresholds (handled by child specs).
```

### Mid-level (engineer level, child of above)

```md
Summary: Rate-limit keying scheme: how API requests map to buckets.

## Guarantees
- Caller can inspect their current bucket key via a debug response header.
- Each request maps to exactly one bucket key.
- Bucket key is stable across retries and pagination (same request params = same bucket).
- Unauthenticated requests use IP-based fallback bucket.

## Constraints
- Bucket key must be deterministic — no randomness or time-based jitter in classification.

## Non-goals
- Bucket capacity/threshold configuration (separate spec).
```

### Leaf (detailed, child of above)

```md
Summary: Unauthenticated request rate-limit bucket uses client IP with /24 prefix grouping.

## Guarantees
- Unauthenticated requests are bucketed by IPv4 /24 prefix (or IPv6 /48).
- Bucket capacity: 100 requests per minute per prefix.
- When multiple proxies are present, the leftmost non-private IP in X-Forwarded-For is used.

## Constraints
- Private/reserved IPs (10.x, 172.16-31.x, 192.168.x) are never used as bucket keys.
- Missing or unparseable IP falls back to a global "unknown" bucket with reduced capacity (10 req/min).

## Rationale
- /24 grouping prevents per-IP evasion while avoiding over-blocking shared NATs.
- Leftmost non-private IP is the standard proxy-aware approach; rightmost is spoofable.
```
