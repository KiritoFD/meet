import asyncio
from functools import wraps

def with_retry(max_retries: int = 3, delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    await asyncio.sleep(delay * (attempt + 1))
            raise last_error
        return wrapper
    return decorator 