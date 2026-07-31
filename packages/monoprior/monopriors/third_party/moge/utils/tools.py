from typing import *
import time


class timeit:
    _history: Dict[str, List['timeit']] = {}

    def __init__(self, name: str = None, verbose: bool = True, average: bool = False):
        self.name = name
        self.verbose = verbose
        self.start = None
        self.end = None
        self.average = average
        if average and name not in timeit._history:
            timeit._history[name] = []

    def __call__(self, func: Callable):
        import inspect
        if inspect.iscoroutinefunction(func):
            async def wrapper(*args, **kwargs):
                with timeit(self.name or func.__qualname__):
                    ret = await func(*args, **kwargs)
                return ret
            return wrapper
        else:
            def wrapper(*args, **kwargs):
                with timeit(self.name or func.__qualname__):
                    ret = func(*args, **kwargs)
                return ret
            return wrapper
        
    def __enter__(self):
        self.start = time.time()
        return self

    @property
    def time(self) -> float:
        assert self.start is not None, "Time not yet started."
        assert self.end is not None, "Time not yet ended."
        return self.end - self.start

    @property
    def average_time(self) -> float:
        assert self.average, "Average time not available."
        return sum(t.time for t in timeit._history[self.name]) / len(timeit._history[self.name])

    @property
    def history(self) -> List['timeit']:
        return timeit._history.get(self.name, [])

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        if self.average:
            timeit._history[self.name].append(self)
        if self.verbose:
            if self.average:
                avg = self.average_time
                print(f"{self.name or 'It'} took {avg:.6f} seconds in average.")
            else:
                print(f"{self.name or 'It'} took {self.time:.6f} seconds.")
