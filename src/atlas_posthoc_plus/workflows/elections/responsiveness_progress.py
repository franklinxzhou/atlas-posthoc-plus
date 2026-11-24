"""Flushed stage timing and heartbeats for batch CEAH jobs."""

from __future__ import annotations

from contextlib import contextmanager
from threading import Event, Thread
from time import perf_counter, process_time


class Progress:
    def __init__(self, interval=60.0):
        self.interval = interval
        self.started = perf_counter()

    def log(self, message):
        print(f"[CEAH +{perf_counter() - self.started:.1f}s] {message}", flush=True)

    @contextmanager
    def stage(self, label, *, heartbeat=True):
        start, cpu = perf_counter(), process_time()
        self.log(f"Starting {label}")

        def report(status):
            self.log(f"{status} {label} "
                     f"(wall {perf_counter() - start:.1f}s, CPU {process_time() - cpu:.1f}s)")

        finished = Event()

        def tick():
            while not finished.wait(self.interval):
                report("Still running")

        worker = None
        if heartbeat and self.interval > 0:
            worker = Thread(target=tick, daemon=True)
            worker.start()
        try:
            yield
        except BaseException:
            report("Failed")
            raise
        else:
            report("Finished")
        finally:
            finished.set()
            if worker is not None:
                worker.join()
