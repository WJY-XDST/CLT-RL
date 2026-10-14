"""Focused Tk keyboard input for the passive MuJoCo viewer (replay only).

The passive viewer exposes key presses, not releases or focus loss. A separate
window supplies those events without a global keyboard hook. The shared state
machine imports no Isaac Sim packages and never injects actions or forces.
"""

import importlib.util
import sys
from pathlib import Path


STATE_SOURCE = Path(__file__).resolve().parents[1] / "scripts/rsl_rl/keyboard_control.py"
_spec = importlib.util.spec_from_file_location("sim2sim_keyboard_state", STATE_SOURCE)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module  # dataclasses resolves the defining module.
_spec.loader.exec_module(_module)
KeyboardConfig = _module.KeyboardConfig
KeyboardCommandState = _module.KeyboardCommandState


class KeyboardInput:
    """Debounce X11 auto-repeat release/press pairs and stop on focus loss.

    ``scheduler`` supplies Tk's after_idle/after_cancel, or a test substitute.
    Keep physical holds separate from state.pressed: after Esc/SPACE/stop, an
    auto-repeat from a still-held W must not re-arm motion. On focus loss the
    same rule requires releasing held keys before pressing them again.
    """

    def __init__(self, state, scheduler):
        self.state, self.scheduler = state, scheduler
        self.held = set()
        self.pending_releases = {}
        self.closed = False

    def press(self, key, modifiers=0):
        key = key.upper()
        if self.closed or key not in self.state.KEYS:
            return False
        # X11 Ctrl, Alt/Mod1, Super/Mod4, AltGr/Mod5; allow Shift/Caps/NumLock.
        if modifiers & (0x4 | 0x8 | 0x40 | 0x80):
            return False
        pending = self.pending_releases.pop(key, None)
        if pending is not None:
            self.scheduler.after_cancel(pending)
        if key not in self.held:
            self.held.add(key)
            self.state.handle_key(key, "press")
        return True

    def release(self, key):
        key = key.upper()
        if self.closed or key not in self.state.KEYS:
            return False
        if key not in self.pending_releases:
            self.pending_releases[key] = self.scheduler.after_idle(
                lambda: self._finish_release(key))
        return True

    def _finish_release(self, key):
        self.pending_releases.pop(key, None)
        self.held.discard(key)
        self.state.handle_key(key, "release")

    def stop(self):
        self.state.stop()

    def focus_lost(self):
        # An untouched panel must not take over a fixed/cycle validation run.
        if self.state.active:
            self.stop()
        for key, token in list(self.pending_releases.items()):
            self.scheduler.after_cancel(token)
            self._finish_release(key)

    def close(self):
        self.focus_lost()
        self.closed = True
        self.held.clear()


class MujocoKeyboardController:
    """Pump the focused control window on the same thread as policy inference."""

    def __init__(self, config):
        import tkinter as tk
        from tkinter import ttk

        self.state = KeyboardCommandState(config)
        self.closed = False
        self._tk = tk
        try:
            self.window = tk.Tk()
        except tk.TclError as error:
            raise RuntimeError("Cannot open keyboard panel; use --no-keyboard or --headless") from error
        self.input = KeyboardInput(self.state, self.window)
        self.window.title("MuJoCo robot keyboard commands")
        self.window.geometry("600x305")
        self.window.resizable(False, False)
        frame = ttk.Frame(self.window, padding=12)
        frame.pack(fill="both", expand=True)
        for text in (
            "点击此控制窗口再按键 / Click THIS window to use the keyboard",
            f"W/S: 前后 ±{config.speed:g} m/s     A/D: 转向 ±{config.yaw_rate:g} rad/s",
            f"Q/E: 降低/提高高度，单击 {config.height_step * 1000:g} mm，长按连续调整",
            "松开 WASD: 目标归零    L / Esc: 停止指令    失焦: 停止指令",
            "SPACE: 停止 + 跳跃请求（当前平地策略不执行跳跃）",
        ):
            ttk.Label(frame, text=text).pack(anchor="w", pady=3)
        self.status = tk.StringVar(value="尚未接管：固定/周期命令继续；首次控制按键后进入手动模式")
        self.height_status = tk.StringVar(value=f"高度范围 {config.height_min:.3f}–{config.height_max:.3f} m")
        self.measured_status = tk.StringVar(value="等待策略与仿真状态")
        for variable in (self.status, self.height_status, self.measured_status):
            ttk.Label(frame, textvariable=variable).pack(anchor="w", pady=3)
        buttons = ttk.Frame(frame)
        buttons.pack(anchor="w", pady=6)
        ttk.Button(buttons, text="停止指令 / Stop", command=self.input.stop).pack(side="left", padx=3)
        ttk.Button(buttons, text="初始高度 / Nominal", command=self._nominal_height).pack(side="left", padx=3)
        self.window.bind("<KeyPress>", self._press)
        self.window.bind("<KeyRelease>", self._release)
        self.window.bind("<FocusOut>", self._focus_out)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        print("Keyboard panel ready: click 'MuJoCo robot keyboard commands'; "
              "WASD / Q,E / L,Esc / SPACE. Closing the panel after takeover keeps zero motion targets.", flush=True)

    @staticmethod
    def _key(event):
        return str(event.keysym).upper()

    def _press(self, event):
        if self.input.press(self._key(event), event.state):
            return "break"

    def _release(self, event):
        if self.input.release(self._key(event)):
            return "break"

    def _focus_out(self, _event):
        # Moving focus from the root to an internal button is not app focus loss.
        if not self.closed:
            self.window.after_idle(self._check_focus)

    def _check_focus(self):
        if not self.closed and self.window.focus_displayof() is None:
            self.input.focus_lost()

    def _nominal_height(self):
        self.input.stop()
        self.state.set_height_target(self.state.config.initial_height)

    def advance(self, dt, fallback_height):
        if not self.state.active:
            # Adopt a running demo's current height at the instant of takeover.
            self.state.set_height_target(float(fallback_height))
            self.state.height = self.state.target_height
        if not self.closed:
            try:
                self.window.update()
            except self._tk.TclError:
                self.close()
        command = self.state.advance(dt)
        if self.state.take_jump_request():
            print("SPACE: jump request recorded, NOT executed by this flat 27-D policy.", flush=True)
        return command

    def show_state(self, command, velocity, height, pitch_deg, roll_deg):
        if self.closed:
            return
        mode = "手动控制" if self.state.active else "固定/周期命令（按控制键接管）"
        self.status.set(f"{mode}  vx={command[0]:+.2f} m/s  yaw={command[1]:+.2f} rad/s"
                        f"  跳跃请求={self.state.jump_requests}（不执行）")
        self.height_status.set(f"高度目标={self.state.target_height:.4f} m  "
                               f"策略命令={command[2]:.4f} m  实际={height:.4f} m")
        self.measured_status.set(f"实际 vx={velocity:+.3f} m/s  pitch={pitch_deg:+.1f}°  roll={roll_deg:+.1f}°")

    def close(self):
        if self.closed:
            return
        self.input.close()
        self.closed = True
        try:
            self.window.destroy()
        except self._tk.TclError:
            pass
