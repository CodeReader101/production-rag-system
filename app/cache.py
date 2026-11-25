import hashlib
from typing import Optional

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from monitoring import get_logger

logger = get_logger()


class RedisResponseCache:
    """Response cache backed by Redis."""

    def __init__(
        self,
        ttl_seconds: int = 300,
        url: str = "redis://localhost:6379/0",
        prefix: str = "rcache",
    ):
        self.ttl = ttl_seconds
        self.prefix = prefix
        self._r = aioredis.Redis.from_url(
            url,
            decode_responses=True,
            socket_timeout=1,
            socket_connect_timeout=1,
        )

    def _make_key(self, query: str) -> str:
        normalized = query.lower().strip()
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        return f"{self.prefix}:resp:{digest}"

    async def get(self, query: str) -> Optional[str]:
        try:
            value = await self._r.get(self._make_key(query))
            await self._r.incr(f"{self.prefix}:hits" if value is not None
                               else f"{self.prefix}:misses")
            return value
        except RedisError as e:
            logger.warning(f"Redis get failed, treating as cache miss: {e}")
            return None

    async def set(self, query: str, response: str) -> None:
        try:
            await self._r.set(self._make_key(query), response, ex=self.ttl)
        except RedisError as e:
            logger.warning(f"Redis set failed: {e}")

    async def invalidate(self) -> int:
        """Delete all cached responses (call after the corpus changes)."""
        deleted = 0
        try:
            async for key in self._r.scan_iter(f"{self.prefix}:resp:*"):
                await self._r.delete(key)
                deleted += 1
        except RedisError as e:
            logger.warning(f"Redis invalidate failed: {e}")
        return deleted

    async def get_stats(self) -> dict:
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