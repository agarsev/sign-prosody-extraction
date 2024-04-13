from joblib import Memory
import os

if not os.getenv("NO_CACHE", False):
    memory = Memory(".joblib", verbose=0)
    def cache(func):
        return memory.cache(func)
else:
    def cache(func):
        return func
