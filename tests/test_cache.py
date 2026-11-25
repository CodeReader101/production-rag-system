import pytest
from redis.exceptions import RedisError

from cache import RedisResponseCache


class FakeRedis:
    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = str(value)

    async def incr(self, key):
        self.store[key] = str(int(self.store.get(key, 0)) + 1)

    async def delete(self, key):
        self.store.pop(key, None)

    async def scan_iter(self, pattern):
        prefix = pattern.rstrip("*")
        for key in list(self.store):
            if key.startswith(prefix):
                yield key

    async def aclose(self):
        pass


class BrokenRedis:
    async def get(self, *a, **k): raise RedisError("down")
    async def set(self, *a, **k): raise RedisError("down")
    async def incr(self, *a, **k): raise RedisError("down")
    def scan_iter(self, *a, **k): raise RedisError("down")


@pytest.fixture
def cache():
    c = RedisResponseCache(ttl_seconds=10)
    c._r = FakeRedis()
    return c


async def test_miss_returns_none(cache):
    assert await cache.get("never stored") is None


async def test_set_then_get(cache):
    await cache.set("What are the support hours?", "9 to 6")
    assert await cache.get("What are the support hours?") == "9 to 6"


async def test_key_is_normalized(cache):
    await cache.set("What are the support hours?", "9 to 6")
    assert await cache.get("  WHAT ARE THE SUPPORT HOURS?  ") == "9 to 6"


async def test_stats_track_hits_and_misses(cache):
    await cache.set("q", "a")
    await cache.get("q")
    await cache.get("other")
    stats = await cache.get_stats()
    assert stats["hits"] == 1 and stats["misses"] == 1 and stats["cached_entries"] == 1


async def test_invalidate_clears_responses(cache):
    await cache.set("q1", "a")
    await cache.set("q2", "b")
    assert await cache.invalidate() == 2
    assert await cache.get("q1") is None


async def test_redis_outage_fails_open():
    c = RedisResponseCache()
    c._r = BrokenRedis()
    assert await c.get("q") is None          # treated as a miss
    await c.set("q", "a")                    # must not raise
    assert (await c.get_stats())["hits"] == 0
    assert await c.invalidate() == 0