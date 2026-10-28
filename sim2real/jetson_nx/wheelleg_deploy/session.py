"""Fail-latched obs-to-reference session. No direct motor/current output."""

import math
import time
from .targets import action_to_targets, clip_action
from .schema import validate_observation


class PolicySession:
    def __init__(self, policy, *, clock=time.monotonic, feedback_timeout_s=.1, inference_deadline_s=.02):
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value <= 0
               for value in (feedback_timeout_s, inference_deadline_s)):
            raise ValueError("Timeouts must be positive and finite")
        self.policy, self.clock = policy, clock
        self.feedback_timeout_s, self.inference_deadline_s = feedback_timeout_s, inference_deadline_s
        self.last_seq = self.last_device_time = self.last_received_at = None
        self.started_at = clock()
        self.fault_reason = None

    def fault(self, reason):
        self.fault_reason = self.fault_reason or str(reason)
        # Zero normalized actions mean nonzero leg/height references, NOT a
        # motor stop. Request the MCU's validated fallback instead.
        return dict(kind="fallback_request", reason=self.fault_reason, obs_seq=self.last_seq,
                    motor_enable=False, requires_mcu_verified_fallback=True)

    def check_timeout(self):
        if self.fault_reason:
            return self.fault(self.fault_reason)
        previous = self.started_at if self.last_received_at is None else self.last_received_at
        if self.clock() - previous > self.feedback_timeout_s:
            return self.fault("Observation timeout")
        return None

    def handle(self, message):
        if self.fault_reason:
            return self.fault(self.fault_reason)
        try:
            timeout = self.check_timeout()
            if timeout:
                return timeout
            if not isinstance(message, dict) or message.get("kind") != "observation":
                raise ValueError("Expected an observation packet")
            if message.get("schema_sha256") != self.policy.schema["sha256"]:
                raise ValueError("MCU observation schema differs from selected policy")
            seq, tick = message.get("seq"), message.get("device_time_us")
            for value in (seq, tick):
                if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**64:
                    raise ValueError("Sequence/timestamp must be unsigned 64-bit integers")
            if self.last_seq is not None:
                if seq <= self.last_seq or tick <= self.last_device_time:
                    raise ValueError("Duplicate/out-of-order observation or MCU clock restart")
                if (tick-self.last_device_time)/1e6 > self.feedback_timeout_s:
                    raise ValueError("Device sampling gap exceeds timeout")
            obs = validate_observation(message.get("obs"), self.policy.schema, self.policy.cfg)
            started = self.clock()
            raw = self.policy(obs)
            elapsed = self.clock() - started
            if elapsed < 0 or elapsed > self.inference_deadline_s:
                raise ValueError("Policy inference deadline missed")
            targets = action_to_targets(raw, obs, self.policy.cfg)
            self.last_seq, self.last_device_time, self.last_received_at = seq, tick, self.clock()
            return dict(kind="reference_request", obs_seq=seq, schema_sha256=self.policy.schema["sha256"],
                        raw_action=raw, clipped_action=clip_action(raw, self.policy.cfg),
                        targets=targets, inference_ms=elapsed*1000, motor_enable=False,
                        offline_only=True, execution_contract=self.policy.metadata["execution_contract"])
        except (ValueError, KeyError, TypeError, RuntimeError) as error:
            return self.fault(error)
