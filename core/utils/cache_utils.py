from typing import Optional, Any, List
from collections import OrderedDict
import time
import threading

from core.config.logging_config import log_info, log_error

# -----------------------------
# Simple LRU cache for results (with optional TTL)
# -----------------------------
class ResultCache:
    """
    Simple thread-safe LRU cache for generation results.
    Key: hashable (e.g. (model_id, prompt, kwargs_as_tuple))
    Value: (timestamp, result)
    """
    def __init__(self, maxsize: int = 1024, ttl: Optional[int] = None):
        self.maxsize = maxsize
        self.ttl = ttl  # seconds
        self.lock = threading.Lock()
        self._cache = OrderedDict()

    def _is_expired(self, entry_time: float) -> bool:
        return self.ttl is not None and (time.time() - entry_time) > self.ttl

    def get(self, key):
        with self.lock:
            if key not in self._cache:
                return None
            entry_time, value = self._cache.pop(key)
            if self._is_expired(entry_time):
                # expired
                return None
            # move to end (most recently used)
            self._cache[key] = (entry_time, value)
            return value

    def set(self, key, value):
        with self.lock:
            if key in self._cache:
                self._cache.pop(key)
            self._cache[key] = (time.time(), value)
            # evict oldest if needed
            while len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    def clear(self):
        with self.lock:
            self._cache.clear()


# -----------------------------
# LRU Manager for loaded models (stores instances)
# -----------------------------
class ModelLRUStore:
    """
    Keep loaded model instances with LRU eviction.
    Stores objects like HuggingFaceCausalLM instances.
    Thread-safe.
    """
    def __init__(self, max_models: int = 3):
        self.max_models = max_models
        self.lock = threading.Lock()
        self._store: "OrderedDict[str, Any]" = OrderedDict()

    def get(self, model_id: str):
        with self.lock:
            if model_id not in self._store:
                return None
            # move to end -> most recently used
            obj = self._store.pop(model_id)
            self._store[model_id] = obj
            return obj

    def put(self, model_id: str, obj: Any):
        with self.lock:
            if model_id in self._store:
                # replace and mark as recent
                self._store.pop(model_id)
            self._store[model_id] = obj
            # evict if exceed
            while len(self._store) > self.max_models:
                evicted_id, evicted_obj = self._store.popitem(last=False)
                log_info(f"LRU: evicting model {evicted_id}")
                # try to cleanup evicted object if it has cleanup method
                try:
                    if hasattr(evicted_obj, "unload"):
                        evicted_obj.unload()
                    elif hasattr(evicted_obj, "clear_cache"):
                        evicted_obj.clear_cache()
                except Exception as e:
                    log_error(f"Error while cleaning evicted model {evicted_id}: {e}")

    def remove(self, model_id: str):
        with self.lock:
            return self._store.pop(model_id, None)

    def list_models(self) -> List[str]:
        with self.lock:
            return list(self._store.keys())

    def clear_all(self):
        with self.lock:
            for model_id, obj in self._store.items():
                try:
                    if hasattr(obj, "unload"):
                        obj.unload()
                    elif hasattr(obj, "clear_cache"):
                        obj.clear_cache()
                except Exception as e:
                    log_error(f"Error clearing model {model_id}: {e}")
            self._store.clear()
