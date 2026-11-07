import hashlib
from typing import Optional

import redis

class RedisResponseCache:
    """Response cache backed by Redis, with the same interface as ResponseCache."""

    def __init__(
        self,
        ttl_seconds: int = 300,
        url: str = "redis://localhost:6379/0",
        prefix: str = "rcache",
    ):
        self.ttl = ttl_seconds
        self.prefix = prefix
        self._r = redis.Redis.from_url(
            url,
            decode_responses=True,    # return str instead of bytes
            socket_timeout=1,         # don't let a slow Redis hang your app
            socket_connect_timeout=1,
        )

    def _make_key(self, query: str) -> str:
        """create a cache key from the normalized query"""
        normalized = query.lower().strip()
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        return f"{self.prefix}:resp:{digest}"

    def get(self, query: str) -> Optional[str]:
        """
        Get cached response if it exists and hasn't expired.
        Returns None on cache miss.
        """
        try:
            value = self._r.get(self._make_key(query))
            # Redis already dropped expired keys, so no TTL check needed
            self._r.incr(f"{self.prefix}:hits" if value is not None
                         else f"{self.prefix}:misses")
            return value
        except redis.RedisError:
            return None  # fail open: treat Redis outages as a cache miss

    def set(self, query: str, response: str) -> None:
        """Cache a response"""
        try:
            # ex=... sets the value and TTL atomically
            self._r.set(self._make_key(query), response, ex=self.ttl)
        except redis.RedisError:
            pass  # caching is best-effort; never break the main request

    @property
    def stats(self) -> dict:
        """Cache performance statistics."""
        try:
            hits = int(self._r.get(f"{self.prefix}:hits") or 0)
            misses = int(self._r.get(f"{self.prefix}:misses") or 0)
            entries = sum(1 for _ in self._r.scan_iter(f"{self.prefix}:resp:*"))
        except redis.RedisError:
            return {"hits": 0, "misses": 0, "hit_rate": 0.0, "cached_entries": 0}

        total = hits + misses
        return {
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / total if total else 0.0,
            "cached_entries": entries,
        }