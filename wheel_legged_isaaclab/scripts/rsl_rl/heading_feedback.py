"""Optional navigation outer loop; does not change the learned rate policy."""

import math


class HeadingFeedback:
    def __init__(self, kp=1.5, ki=0.5, rate_limit=0.5, integral_limit=0.1):
        if not all(math.isfinite(v) and v > 0 for v in (kp, ki, rate_limit, integral_limit)):
            raise ValueError("Heading gains and limits must be finite and positive")
        self.kp, self.ki = kp, ki
        self.rate_limit, self.integral_limit = rate_limit, integral_limit
        self.reset()

    def reset(self):
        self.reference = 0.0
        self.integral_rate = 0.0

    def step(self, requested_rate, measured_turn, dt):
        if not all(math.isfinite(v) for v in (requested_rate, measured_turn, dt)) or dt <= 0:
            raise ValueError("Invalid heading feedback sample")
        if abs(requested_rate) > self.rate_limit:
            raise ValueError("Requested yaw rate exceeds the learned range")
        error = math.atan2(math.sin(self.reference - measured_turn), math.cos(self.reference - measured_turn))
        integral = max(-self.integral_limit, min(self.integral_limit, self.integral_rate + self.ki * error * dt))
        proposed = requested_rate + self.kp * error + integral
        # Integrate only if unsaturated, or if the error brings the output back into range.
        if abs(proposed) <= self.rate_limit or proposed * error < 0:
            self.integral_rate = integral
        output = requested_rate + self.kp * error + self.integral_rate
        # Advance the user's reference only, never integrate our own correction.
        self.reference += requested_rate * dt
        return max(-self.rate_limit, min(self.rate_limit, output))
