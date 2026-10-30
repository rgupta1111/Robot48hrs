"""Threads that keep camera frames, detection and arm moves from waiting on each other.

FrameGrabber  several overlapping camera requests -> newest frame (network latency stops limiting the frame rate)
Perception    runs the detector on every new frame -> latest result (frame, features, fps)
MoveWorker    executes the arm moves one at a time off the control loop, so the loop never blocks on the arm
"""
import threading
import time
from dataclasses import dataclass, field


class _Rate:
    """Events per second over the last 2 seconds (steady even when frames arrive in bursts)."""

    def __init__(self):
        self._times = []

    def tick(self):
        now = time.monotonic()
        self._times = [t for t in self._times if now - t < 2.0] + [now]

    @property
    def value(self):
        now = time.monotonic()
        recent = [t for t in self._times if now - t < 2.0]
        if len(recent) < 2:
            return 0.0
        return (len(recent) - 1) / max(recent[-1] - recent[0], 1e-6)


class FrameGrabber:
    def __init__(self, fetch, workers=4):
        self.fetch, self.workers = fetch, workers
        self._mu, self._frame, self._t, self.seq, self.fps = threading.Lock(), None, 0.0, 0, _Rate()
        self._alive, self._threads = True, []

    def start(self):
        for i in range(self.workers):
            t = threading.Thread(target=self._run, args=(i * 0.08,), daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def _run(self, delay=0.0):
        time.sleep(delay)                        # stagger the workers so frames arrive evenly, not in bursts
        while self._alive:
            t0 = time.monotonic()
            try:
                frame = self.fetch()
            except Exception as e:  # keep grabbing
                print("camera error:", e, flush=True)
                time.sleep(0.5)
                continue
            if frame is None:
                continue
            t = (t0 + time.monotonic()) / 2      # the picture was taken somewhere during the request
            with self._mu:
                if t > self._t:                  # never go back to an older frame
                    self._frame, self._t = frame, t
                    self.seq += 1
                    self.fps.tick()

    def get(self):
        with self._mu:
            return self._frame, self._t, self.seq

    def stop(self):
        self._alive = False


@dataclass
class Result:
    frame: object
    t: float                      # when the frame was captured
    feats: object = None          # (u, v, h) or None
    live: bool = False
    dets: list = field(default_factory=list)
    done: float = 0.0             # when detection finished


class Perception(threading.Thread):
    def __init__(self, grabber, analyse):
        """analyse(frame) -> (result fields dict); it runs on this thread and may use self.mu to protect shared state."""
        super().__init__(daemon=True)
        self.grabber, self.analyse = grabber, analyse
        self.mu, self.result, self.fps, self._alive = threading.Lock(), None, _Rate(), True

    def run(self):
        last = -1
        while self._alive:
            frame, t, seq = self.grabber.get()
            if frame is None or seq == last:
                time.sleep(0.002)
                continue
            last = seq
            with self.mu:
                fields = self.analyse(frame)
            self.result = Result(frame=frame, t=t, done=time.monotonic(), **fields)
            self.fps.tick()

    def stop(self):
        self._alive = False


class MoveWorker(threading.Thread):
    """One arm move at a time. submit() is refused while busy, so commands never pile up behind a slow move."""

    def __init__(self, do_move):
        super().__init__(daemon=True)
        self.do_move, self.cv = do_move, threading.Condition()
        self._job, self.busy, self.end_t, self.error, self._alive = None, False, 0.0, None, True

    def submit(self, job):
        with self.cv:
            if self.busy:
                return False
            self._job, self.busy = job, True
            self.cv.notify()
            return True

    def run(self):
        while self._alive:
            with self.cv:
                while self._job is None and self._alive:
                    self.cv.wait(0.2)
                job, self._job = self._job, None
            if job is None:
                continue
            try:
                self.do_move(job)
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
            self.end_t = time.monotonic()
            self.busy = False

    def stop(self):
        self._alive = False
        with self.cv:
            self.cv.notify()
