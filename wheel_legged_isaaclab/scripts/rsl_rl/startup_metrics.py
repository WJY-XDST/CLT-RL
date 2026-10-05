"""Capture a startup transient independently of rolling dashboard histories."""

import math
import statistics

from analyze_response import settling_time


class StartupCapture:
    def __init__(self, dt, duration=5.0):
        self.dt = dt
        self.count = round(duration / dt)
        self.samples = []

    @property
    def complete(self):
        return len(self.samples) == self.count

    def add(self, left_rad, right_rad):
        if not all(math.isfinite(v) for v in (left_rad, right_rad)):
            raise ValueError("Startup angles must be finite")
        if not self.complete:
            self.samples.append((math.degrees(left_rad), math.degrees(right_rad)))

    def report(self):
        if not self.complete:
            return None
        result = {}
        for index, side in enumerate(("left", "right")):
            values = [row[index] for row in self.samples]
            steady = statistics.mean(values[-round(2.0 / self.dt):])
            result[side] = {
                "steady_deg": steady,
                "peak_abs_deg": max(map(abs, values)),
                "settling_s": settling_time([abs(v - steady) for v in values], .5, self.dt),
                "variation_first_2s_deg": sum(abs(values[i] - values[i - 1])
                                               for i in range(1, round(2.0 / self.dt))),
            }
        return result
