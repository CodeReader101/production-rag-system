import hashlib
from typing import Optional

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from config import get_settings
class RedisResponseCache:
    """Response cache backed by Redis, with the same interface as ResponseCache."""

    def __init__(
        self,
        prefix: str = "rcache",
    ):
        settings = get_settings()
        self.ttl = settings.cache_ttl_seconds
        self.prefix = prefix
        self._r = aioredis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,    # return str instead of bytes
            socket_timeout=1,         # don't let a slow Redis hang your app
            socket_connect_timeout=1,
        )

    def _make_key(self, query: str) -> str:
        """create a cache key from the normalized query"""
        normalized = query.lower().strip()
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        return f"{self.prefix}:resp:{digest}"

    async def get(self, query: str) -> Optional[str]:
        """
        Get cached response if it exists and hasn't expired.
        Returns None on cache miss.
        """
        try:
            value = await self._r.get(self._make_key(query))
            # Redis already dropped expired keys, so no TTL check needed
            await self._r.incr(f"{self.prefix}:hits" if value is not None
                         else f"{self.prefix}:misses")
            return value
        except RedisError:
            return None  # fail open: treat Redis outages as a cache miss

    async def set(self, query: str, response: str) -> None:
        """Cache a response"""
        try:
            # ex=... sets the value and TTL atomically
            await self._r.set(self._make_key(query), response, ex=self.ttl)
        except RedisError:
            pass  # caching is best-effort; never break the main request

    async def get_stats(self) -> dict:
        """Cache performance statistics."""
        try:
            hits = int(await self._r.get(f"{self.prefix}:hits") or 0)
            misses = int(await self._r.get(f"{self.prefix}:misses") or 0)
            entries = 0
            async for _ in self._r.scan_iter(f"{self.prefix}:resp:*"):
                entries += 1
        except RedisError:
            return {"hits": 0, "misses": 0, "hit_rate": 0.0, "cached_entries": 0}

        total = hits + misses
        return {
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / total if total else 0.0,
            "cached_entries": entries,
        }

    async def close(self):
        await self._r.aclose()   