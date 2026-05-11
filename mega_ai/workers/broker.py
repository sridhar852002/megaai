import os

import dramatiq
from dramatiq.brokers.redis import RedisBroker

redis_broker = RedisBroker(url=os.environ["REDIS_URL"])  # type: ignore[no-untyped-call]
dramatiq.set_broker(redis_broker)
