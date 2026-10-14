"""Replay-only keyboard commands; importing this module never starts Isaac Sim.

W/S: forward/back, A/D: yaw left/right, Q/E: lower/raise body-height
command (indirectly shorten/extend legs). SPACE: stop and queue a jump only
when the loaded task exposes a jump handler. No keyboard force is injected.
"""
from dataclasses import dataclass
import math


def supports_keyboard_task(task_name):
    """Both registered flat tasks share the same three command observations."""
    return task_name in ('WheelLeggedVMC-Flat-v0', 'WheelLeggedJointPD-Flat-v0')


@dataclass(frozen=True)
class KeyboardConfig:
    speed: float = 0.8
    yaw_rate: float = 0.6
    height_rate: float = 0.02
    height_min: float = 0.28
    height_max: float = 0.32
    initial_height: float = 0.30
    height_step: float = 0.005
    height_slew_rate: float = 0.02

    def __post_init__(self):
        if not all(math.isfinite(v) for v in self.__dict__.values()):
            raise ValueError('Keyboard settings must be finite')
        if min(self.speed, self.yaw_rate, self.height_rate, self.height_min,
               self.height_step, self.height_slew_rate) <= 0:
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
        self.target_height = config.initial_height
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
                if key in ('Q', 'E') and ({'Q', 'E'} - {key}).isdisjoint(self.pressed):
                    # Change the user's goal, not the command presented to the
                    # policy. Repeats do not add steps; advance() slews to it.
                    direction = 1 if key == 'E' else -1
                    self.set_height_target(self.target_height + direction * self.config.height_step)
        # OS KEY_REPEAT must not accumulate speed, height steps or jumps.
        return True

    def stop(self):
        self.pressed.clear()
        self.active = True

    def set_height_target(self, height):
        if not math.isfinite(height):
            raise ValueError('Keyboard height target must be finite')
        self.target_height = min(self.config.height_max, max(self.config.height_min, height))

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
        self.set_height_target(self.target_height +
            (int('E' in self.pressed) - int('Q' in self.pressed)) * c.height_rate * dt)
        delta = self.target_height - self.height
        max_change = c.height_slew_rate * dt
        if abs(delta) <= max_change:
            self.height = self.target_height
        else:
            self.height += math.copysign(max_change, delta)
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
    if hasattr(base_env.cfg, 'jump'):
        # The separate stationary jump policy has no locomotion training.
        # Never feed nonzero WASD targets into its 39-D observation.
        speed, yaw = 0., 0.
        command = (speed, yaw, height)
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
    def __init__(self, config, active=False, jump_handler=None):
        import carb.input
        import omni.appwindow
        import omni.ui as ui
        self.state = KeyboardCommandState(config, active)
        self._jump_handler = jump_handler
        self._carb = carb.input
        self._input = carb.input.acquire_input_interface()
        self._app_window = omni.appwindow.get_default_app_window()
        if self._app_window is None or self._app_window.get_keyboard() is None:
            raise RuntimeError('No GUI keyboard available; use --no_keyboard for automated replay')
        self._keyboard = self._app_window.get_keyboard()
        self._subscription = None
        self.window = ui.Window('Keyboard robot commands', width=460, height=275,
                                position_x=20, position_y=200)
        with self.window.frame:
            with ui.VStack(spacing=5):
                ui.Label('Click simulator window; hold keys to command motion.', height=22)
                ui.Label('W/S: forward/back    A/D: turn left/right', height=22)
                ui.Label(f'Q/E: height tap {config.height_step * 1000:g} mm; hold '
                         f'{config.height_rate * 1000:g} mm/s', height=22)
                ui.Label('SPACE: stop + jump request' if jump_handler else
                         'SPACE: stop + jump request (unavailable in this task)', height=22)
                ui.Label('Release WASD: zero motion    L / Esc: stop', height=22)
                self.status = ui.Label('Keyboard ready; waiting for input.', height=24)
                self.height_status = ui.Label('Height goal and ramped command ready.', height=22)
                self.jump_status = ui.Label('Jump handler available; use a validated jump checkpoint.' if jump_handler else
                                            'Jump unavailable: no jump command in current policy.', height=22)
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
              'Release motion keys to stop; L/Esc stop. ' + (
                  'SPACE queues the dedicated jump task; proficiency is not guaranteed.' if jump_handler else
                  'SPACE is reserved, NOT a trained jump.'), flush=True)

    def _reset_height(self):
        self.state.stop()
        self.state.set_height_target(self.state.config.initial_height)

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
            if self._jump_handler is not None:
                self._jump_handler()
                self.jump_status.text = f'Jump request #{self.state.jump_requests}: queued; waits for stable stance.'
                print('[KEYBOARD] Jump request sent to the dedicated task; '
                      'no keyboard force, action pulse or teleport injected.', flush=True)
            else:
                self.jump_status.text = f'Jump request #{self.state.jump_requests}: NOT executed (untrained).'
                print('[KEYBOARD] Jump requested; motion target set to zero. '
                      'Current policy has no jump command: request NOT executed. '
                      'No force, action pulse or teleport is injected.', flush=True)
        if command is not None:
            speed, yaw, height = command
            self.status.text = f'vx={speed:+.2f} m/s   yaw={yaw:+.2f} rad/s   height={height:.3f} m'
            self.height_status.text = (f'Height goal {self.state.target_height:.3f} | '
                                       f'ramped cmd {height:.3f} m | '
                                       f'limit {self.state.config.height_slew_rate * 1000:g} mm/s')
            message = (speed, yaw, round(height, 3), round(self.state.target_height, 3))
            if message != self._last_message:
                print('[KEYBOARD] ' + self.status.text +
                      f'   height_goal={self.state.target_height:.3f} m', flush=True)
                self._last_message = message
        return command

    def close(self):
        if self._subscription is not None:
            self._input.unsubscribe_to_input_events(self._subscription)
            self._subscription = None
        if self.window is not None:
            self.window.destroy()
            self.window = None
