"""
Rate limiter for API requests to avoid hitting rate limits.
"""
import asyncio
import time
import logging
from typing import Dict, Optional
import functools

class RateLimiter:
    """
    Rate limiter for API calls.
    
    Ensures that API calls don't exceed a specified rate limit.
    """
    
    def __init__(self, max_calls: int = 750, time_period: int = 60):
        """
        Initialize the rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in the time period
            time_period: Time period in seconds (default: 60 seconds)
        """
        self.max_calls = max_calls
        self.time_period = time_period
        self.calls = []  # Timestamp of each call
        self.semaphore = asyncio.Semaphore(max_calls)  # Limit concurrent requests
        self.lock = asyncio.Lock()  # Lock for thread safety
        self.logger = logging.getLogger(__name__)
        
        # Stats
        self.total_calls = 0
        self.throttled_calls = 0
        
        self.logger.info(f"Rate limiter initialized: {max_calls} calls per {time_period} seconds")
    
    async def wait_for_capacity(self):
        """
        Wait until there is capacity to make a new API call.
        """
        async with self.lock:
            # Remove calls that are outside the time window
            current_time = time.time()
            self.calls = [t for t in self.calls if current_time - t < self.time_period]
            
            # Check if we've hit the rate limit
            if len(self.calls) >= self.max_calls:
                # Calculate how long to wait
                oldest_call = min(self.calls)
                wait_time = self.time_period - (current_time - oldest_call) + 0.1  # Add a small buffer
                
                if wait_time > 0:
                    self.throttled_calls += 1
                    self.logger.warning(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                    # Release the lock while waiting
                    self.lock.release()
                    await asyncio.sleep(wait_time)
                    await self.lock.acquire()
            
            # Add the current call
            self.calls.append(time.time())
            self.total_calls += 1
            
            # Log statistics periodically
            if self.total_calls % 100 == 0:
                self.logger.info(f"API calls: {self.total_calls} total, {self.throttled_calls} throttled")
    
    async def acquire(self):
        """
        Acquire permission to make an API call.
        """
        await self.semaphore.acquire()
        await self.wait_for_capacity()
    
    def release(self):
        """
        Release the semaphore after an API call.
        """
        self.semaphore.release()
    
    def __call__(self, func):
        """
        Decorator for rate-limiting async functions.
        """
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            await self.acquire()
            try:
                return await func(*args, **kwargs)
            finally:
                self.release()
        return wrapper

# Create a global rate limiter instance
global_rate_limiter = RateLimiter(max_calls=750, time_period=60)

# Simple helper function to rate-limit async API calls
async def rate_limited_api_call(func, *args, **kwargs):
    """
    Make a rate-limited API call.
    
    Args:
        func: Async function to call
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        The result of the function call
    """
    await global_rate_limiter.acquire()
    try:
        return await func(*args, **kwargs)
    finally:
        global_rate_limiter.release()