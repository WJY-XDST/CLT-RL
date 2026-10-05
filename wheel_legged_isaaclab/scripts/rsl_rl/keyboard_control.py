"""Replay-only keyboard commands; importing this module never starts Isaac Sim.

W/S: forward/back, A/D: yaw left/right, Q/E: lower/raise body-height
command (indirectly shorten/extend legs). SPACE: stop and request a future
jump policy, currently unsupported. No actions, torques or poses are injected.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class KeyboardConfig:
    speed: float = 0.5
    yaw_rate: float = 0.3
    height_rate: float = 0.01
    height_min: float = 0.28
    height_max: float = 0.32
    initial_height: float = 0.30

    def __post_init__(self):
        if not all(math.isfinite(v) for v in self.__dict__.values()):
            raise ValueError('Keyboard settings must be finite')
        if min(self.speed, self.yaw_rate, self.height_rate, self.height_min) <= 0:
            raise ValueError('Keyboard rates and minimum height must be positive')
        if not self.height_min <= self.initial_height <= self.height_max:
            raise ValueError('Initial keyboard height must lie inside configured height bounds')


class KeyboardCommandState:
    """Pure state machine, independently testable without a GUI or simulator."""
    KEYS = frozenset(('W', 'S', 'A', 'D', 'Q', 'E', 'SPACE', 'ESCAPE', 'L'))

    def __init__(self, config, active=False):
        self.config = config
        self.active = active
        self.pressed = set()
        self.height = config.initial_height
        self.jump_requests = 0
        self.pending_jump = False

    def handle_key(self, key, event_type):
        key = key.upper()
        if key not in self.KEYS:
            return False
        if event_type == 'release':
            self.pressed.discard(key)
        elif event_type == 'press' and key not in self.pressed:
            self.active = True
            if key in ('SPACE', 'ESCAPE', 'L'):
                self.stop()
                if key == 'SPACE':
                    self.jump_requests += 1
                    self.pending_jump = True
                self.pressed.add(key)
            else:
                self.pressed.add(key)
        # OS KEY_REPEAT must not accumulate speed, height steps or jumps.
        return True

    def stop(self):
        self.pressed.clear()
        self.active = True

    def take_jump_request(self):
        pending = self.pending_jump
        self.pending_jump = False
        return pending

    def advance(self, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('Keyboard timestep must be finite and positive')
        if not self.active:
            return None
        c = self.config
        self.height = min(c.height_max, max(c.height_min, self.height +
            (int('E' in self.pressed) - int('Q' in self.pressed)) * c.height_rate * dt))
        return (c.speed * (int('W' in self.pressed) - int('S' in self.pressed)),
                c.yaw_rate * (int('A' in self.pressed) - int('D' in self.pressed)), self.height)


def apply_keyboard_command(base_env, observation, command):
    """Write task commands AND policy input; preserve the env's speed ramp.

    Collapsed ranges prevent periodic command resampling/reset from restoring
    random motion. Does not call _get_observations (which advances history).
    """
    if command is None:
        return
    speed, yaw, height = command
    base_env.cfg.commands.heading_command = False
    for index, value in enumerate(command):
        base_env._command_ranges[index, :] = value
    base_env._target_lin_vel_x[:] = speed
    base_env._commands[:, 1] = yaw
    base_env._commands[:, 2] = height
    policy_obs = observation['policy'] if hasattr(observation, 'keys') and 'policy' in observation.keys() else observation
    policy_obs[:, 6:9] = base_env._commands[:, :3] * base_env._commands_scale


class IsaacKeyboardController:
    """Subscribe inside the replay process; no global OS keyboard hook needed."""
    def __init__(self, config, active=False):
        import carb.input
        import omni.appwindow
        import omni.ui as ui
        self.state = KeyboardCommandState(config, active)
        self._carb = carb.input
        self._input = carb.input.acquire_input_interface()
        self._app_window = omni.appwindow.get_default_app_window()
        if self._app_window is None or self._app_window.get_keyboard() is None:
            raise RuntimeError('No GUI keyboard available; use --no_keyboard for automated replay')
        self._keyboard = self._app_window.get_keyboard()
        self._subscription = None
        self.window = ui.Window('Keyboard robot commands', width=440, height=245,
                                position_x=20, position_y=200)
        with self.window.frame:
            with ui.VStack(spacing=5):
                ui.Label('Click simulator window; hold keys to command motion.', height=22)
                ui.Label('W/S: forward/back    A/D: turn left/right', height=22)
                ui.Label('Q/E: shorten/extend via body-height command', height=22)
                ui.Label('SPACE: stop + jump request (jump NOT trained)', height=22)
                ui.Label('Release WASD: zero motion    L / Esc: stop', height=22)
                self.status = ui.Label('Keyboard ready; waiting for input.', height=24)
                self.jump_status = ui.Label('Jump unavailable: no jump command in current policy.', height=22)
                with ui.HStack(height=28, spacing=5):
                    ui.Button('Stop command', clicked_fn=self.state.stop)
                    ui.Button('Nominal height', clicked_fn=self._reset_height)
        # Subscribe before Kit hotkeys (order=0) and consume only our keys.
        # Otherwise SPACE may pause the timeline and W may change the gizmo.
        try:
            self._subscription = self._input.subscribe_to_input_events(
                self._on_input_event, device=self._keyboard, order=0)
        except Exception:
            self.window.destroy()
            self.window = None
            raise
        self._last_message = None
        print('[KEYBOARD] Automatic controller enabled: WASD, Q/E, SPACE. '
              'Release motion keys to stop; L/Esc stop. SPACE is reserved, NOT a trained jump.', flush=True)

    def _reset_height(self):
        self.state.stop()
        self.state.height = self.state.config.initial_height

    def _on_event(self, event, *args):
        types = self._carb.KeyboardEventType
        kind = {types.KEY_PRESS: 'press', types.KEY_RELEASE: 'release', types.KEY_REPEAT: 'repeat'}.get(event.type)
        modifier_mask = (self._carb.KEYBOARD_MODIFIER_FLAG_CONTROL | self._carb.KEYBOARD_MODIFIER_FLAG_ALT
                         | self._carb.KEYBOARD_MODIFIER_FLAG_SUPER)
        if kind != 'release' and event.modifiers & modifier_mask:
            # Ctrl+S / Alt+W and OS shortcuts are not robot commands. Still
            # accept releases with modifiers so a previously held key clears.
            return True
        if kind is not None and (kind == 'release' or self._app_window.is_focused()):
            return not self.state.handle_key(event.input.name, kind)
        return True

    def _on_input_event(self, event, *args):
        if event.deviceType == self._carb.DeviceType.KEYBOARD:
            return self._on_event(event.event, *args)
        return True

    def advance(self, dt):
        # Window switches can lose key-release events. Never keep driving from
        # a latched W/S/A/D/Q/E when the simulator loses OS window focus.
        if self.state.active and not self._app_window.is_focused():
            self.state.stop()
        command = self.state.advance(dt)
        if self.state.take_jump_request():
            self.jump_status.text = f'Jump request #{self.state.jump_requests}: NOT executed (untrained).'
            print('[KEYBOARD] Jump requested; motion target set to zero. '
                  'Current policy has no jump command: request NOT executed. '
                  'No force, action pulse or teleport is injected.', flush=True)
        if command is not None:
            speed, yaw, height = command
            self.status.text = f'vx={speed:+.2f} m/s   yaw={yaw:+.2f} rad/s   height={height:.3f} m'
            message = (speed, yaw, round(height, 3))
            if message != self._last_message:
                print('[KEYBOARD] ' + self.status.text, flush=True)
                self._last_message = message
        return command

    def close(self):
        if self._subscription is not None:
            self._input.unsubscribe_to_input_events(self._subscription)
            self._subscription = None
        if self.window is not None:
            self.window.destroy()
            self.window = None
