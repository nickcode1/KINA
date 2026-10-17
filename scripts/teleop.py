"""Keyboard teleoperation of BOTH Panda arms in the MuJoCo viewer (20 Hz control) + demo recording.

    .venv\Scripts\python scripts\teleop.py            # teleop only
    .venv\Scripts\python scripts\teleop.py --record   # + save each episode to data/demos/*.hdf5

Keys are read with pynput (held-key state -> smooth velocity commands) and are only accepted while the
MuJoCo viewer window has keyboard focus. The action mapping reuses robosuite's Device logic
(``Device.input2action``, ``active_robot`` / ``active_arm`` switching, per-arm gripper state), exactly like
robosuite's ``collect_human_demonstrations.py``. Each Panda is its own robot (robot0 = RIGHT arm,
robot1 = LEFT arm), so "switch arm" switches ``device.active_robot``.

Key bindings (world frame; robots sit at -x and face the user at +x).  The "alt" column also works with
``--keys viewer`` (MuJoCo key_callback fallback: the viewer swallows letters and Enter there).
    key                  alt          action
    Up / Down arrow                   move +x / -x  (forward towards the user / back)
    Left / Right arrow                move +y / -y  (to the robots' left / right)
    E / Q                PgUp / PgDn  move up / down (+z / -z)
    Z / X                3 / 4        roll  + / -   (about world x)
    C / V                5 / 6        pitch + / -   (about world y)
    A / D                7 / 8        yaw   + / -   (about world z)
    hold Shift                        slow / fine mode (x0.3; pynput only)
    F                    9            toggle gripper of the active arm (open <-> close)
    N                    0            switch active arm (right <-> left)
    1 / 2                             select RIGHT arm / LEFT arm
    Y                    Insert       toggle the episode's "success" flag
    Enter                Home         end episode: SAVE it (with --record) and reset
    Delete               End          end episode: DISCARD it and reset
    Esc  (or close the window)        quit (an unsaved episode is discarded)
The MuJoCo viewer also reacts to some of these keys (visualisation toggles, geom groups); they are undone
every frame. Avoid Space / Backspace / Tab / S W R L K G (viewer pause / reset / panels / render flags).
"""
import argparse
import ctypes
import datetime
import os
import queue
import sys
import threading
import time

import h5py
import numpy as np
from _common import ROOT, make_env
from robosuite.devices import Device
from robosuite.utils.camera_utils import get_real_depth_map

DEMO_DIR = os.path.join(ROOT, "data", "demos")
CAM = "rgbd_cam"
ARM_LABEL = ["RIGHT arm (robot0)", "LEFT arm (robot1)"]

# held keys -> (index into the 6-d [x, y, z, rx, ry, rz] command, sign)
MOVE_KEYS = {
    "up": (0, +1), "down": (0, -1), "left": (1, +1), "right": (1, -1),
    "e": (2, +1), "q": (2, -1), "page_up": (2, +1), "page_down": (2, -1),
    "z": (3, +1), "x": (3, -1), "3": (3, +1), "4": (3, -1),
    "c": (4, +1), "v": (4, -1), "5": (4, +1), "6": (4, -1),
    "a": (5, +1), "d": (5, -1), "7": (5, +1), "8": (5, -1),
}
EVENT_KEYS = {"f": "gripper", "9": "gripper", "n": "switch", "0": "switch", "1": "arm0", "2": "arm1",
              "y": "success", "insert": "success", "enter": "save", "home": "save",
              "delete": "discard", "end": "discard", "esc": "quit"}
SHIFT_KEYS = {"shift", "shift_l", "shift_r"}

KEY_HELP = """
================ BimanualAssist keyboard teleop ================
  Up/Down arrows       +x / -x (towards user / back)
  Left/Right arrows    +y / -y
  E / Q   (PgUp/PgDn)  up / down
  Z / X   (3 / 4)      roll  +/-
  C / V   (5 / 6)      pitch +/-
  A / D   (7 / 8)      yaw   +/-
  hold Shift           slow mode
  F       (9)          toggle gripper (active arm)
  N       (0)          switch arm      1 = RIGHT arm   2 = LEFT arm
  Y       (Insert)     toggle success flag
  Enter   (Home)       save episode (--record) + reset
  Delete  (End)        discard episode + reset
  Esc / close window   quit
 (click the MuJoCo window first - keys only count while it has focus)
================================================================"""
# ----------------------------------------------------------------------------- device
class TeleopKeyboard(Device):
    """robosuite Device driven by a set of currently-held key names (velocity control).

    Key names are lower-case strings: single characters ('e', '1') or special names
    ('up', 'enter', 'shift', ...). ``press``/``release`` may be called from any thread (pynput
    listener, viewer key callback, or a scripted test); discrete events are queued and applied
    in the control thread by ``TeleopSession``.
    """

    def __init__(self, env, pos_speed=0.3, rot_speed=0.3, slow_factor=0.3):
        super().__init__(env)
        self.pos_speed, self.rot_speed, self.slow_factor = pos_speed, rot_speed, slow_factor
        self._lock = threading.Lock()
        self._held = {}  # name -> expiry time (inf = until released)
        self.events = queue.Queue()
        self._reset_internal_state()

    # --- key input
    def press(self, name, hold_for=None):
        if name in EVENT_KEYS:
            with self._lock:
                already = name in self._held
                self._held[name] = np.inf
            if not already:  # ignore OS auto-repeat
                self.events.put(EVENT_KEYS[name])
            return
        with self._lock:
            self._held[name] = np.inf if hold_for is None else time.perf_counter() + hold_for

    def release(self, name):
        with self._lock:
            self._held.pop(name, None)

    def release_all(self):
        with self._lock:
            self._held.clear()

    def held(self):
        now = time.perf_counter()
        with self._lock:
            for k in [k for k, t in self._held.items() if t < now]:
                del self._held[k]
            return set(self._held)

    def command(self):
        """Normalised 6-d command in [-1, 1]: [vx, vy, vz, wx, wy, wz] (world/base frame)."""
        keys = self.held()
        v = np.zeros(6)
        for k in keys:
            if k in MOVE_KEYS:
                i, s = MOVE_KEYS[k]
                v[i] += s
        v = np.clip(v, -1, 1)
        v[:3] *= self.pos_speed
        v[3:] *= self.rot_speed
        if keys & SHIFT_KEYS:
            v *= self.slow_factor
        return v

    # --- robosuite Device interface
    def start_control(self):
        self._reset_internal_state()

    def get_controller_state(self):
        v = self.command()
        # Device.input2action does drot = raw[[1, 0, 2]]; drot[2] *= -1  -> pre-invert that here
        raw_drotation = np.array([v[4], v[3], -v[5]])
        return dict(dpos=v[:3], rotation=np.eye(3), raw_drotation=raw_drotation,
                    grasp=int(self.grasp), reset=0, base_mode=0)

    def _postprocess_device_outputs(self, dpos, drotation):
        return np.clip(dpos, -1, 1), np.clip(drotation, -1, 1)

    # --- helpers
    def gripper_closed(self, robot_idx):
        return bool(self.grasp_states[robot_idx][0])

    def build_action(self):
        """Full env action (14-d): active arm from input2action, other arm holds still, both grippers kept."""
        ac_dict = self.input2action()  # never None (reset handled via events)
        parts = []
        for i, robot in enumerate(self.env.robots):
            d = {}
            for arm in robot.arms:
                if i == self.active_robot:
                    d[arm] = ac_dict[f"{arm}_delta"]
                    d[f"{arm}_gripper"] = ac_dict[f"{arm}_gripper"]
                else:
                    d[arm] = np.zeros(6)
                    d[f"{arm}_gripper"] = np.array([1.0 if self.grasp_states[i][0] else -1.0])
            parts.append(robot.create_action_vector(d))
        return np.concatenate(parts)


# ----------------------------------------------------------------------------- recorder
SKIP_OBS = ("proprio-state", "object-state", "_image", "_depth")


class EpisodeRecorder:
    """Buffers (obs_t, action_t, state_t, images_t) per control step; writes one HDF5 per episode."""

    def __init__(self, env, img_size=128, camera=CAM):
        self.env, self.img_size, self.camera = env, img_size, camera
        self.clear()

    def clear(self):
        self.obs, self.actions, self.states, self.rgb, self.depth = {}, [], [], [], []
        self.active, self.attached, self.rewards = [], [], []

    def __len__(self):
        return len(self.actions)

    def render_cam(self):
        s = self.img_size
        rgb, depth = self.env.sim.render(camera_name=self.camera, width=s, height=s, depth=True)
        depth_m = get_real_depth_map(self.env.sim, depth)
        return rgb[::-1].copy(), depth_m[::-1].astype(np.float32)  # flip OpenGL -> top row first

    def add(self, obs, action, active_robot):
        for k, v in obs.items():
            if not any(t in k for t in SKIP_OBS):
                self.obs.setdefault(k, []).append(np.asarray(v, dtype=np.float64).reshape(-1))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.states.append(self.env.sim.get_state().flatten())
        rgb, depth = self.render_cam()
        self.rgb.append(rgb)
        self.depth.append(depth)
        self.active.append(active_robot)
        self.attached.append([self.env.food_attached(i) for i in range(len(self.env.foods))]
                             if hasattr(self.env, "food_attached") else [])

    def add_reward(self, r):
        self.rewards.append(float(r))

    def save(self, success, out_dir=DEMO_DIR, prefix="demo"):
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"{prefix}_{stamp}.hdf5")
        n = 1
        while os.path.exists(path):
            path = os.path.join(out_dir, f"{prefix}_{stamp}_{n}.hdf5")
            n += 1
        env = self.env
        with h5py.File(path, "w") as f:
            a = f.attrs
            a["env"] = "BimanualAssist"
            a["robots"] = "Panda,Panda"
            a["env_configuration"] = "parallel"
            a["controller"] = "OSC_POSE (delta, base frame)"
            a["action_layout"] = ("[robot0/RIGHT: dx dy dz drx dry drz grip, robot1/LEFT: dx dy dz drx dry drz grip]; "
                                  "normalised [-1,1]; pos 1.0 = 0.05 m, rot 1.0 = 0.5 rad; grip -1 open, +1 close")
            a["control_freq"] = env.control_freq
            a["success"] = bool(success)
            a["num_steps"] = len(self)
            a["date"] = datetime.datetime.now().isoformat(timespec="seconds")
            a["camera"] = self.camera
            a["image_size"] = self.img_size
            a["image_convention"] = "row 0 = top of image (already flipped from OpenGL)"
            a["model_file"] = env.sim.model.get_xml()
            f.create_dataset("actions", data=np.array(self.actions))
            f.create_dataset("states", data=np.array(self.states))
            f.create_dataset("rewards", data=np.array(self.rewards[:len(self)], dtype=np.float32))
            f.create_dataset("active_robot", data=np.array(self.active, dtype=np.int8))
            f.create_dataset("food_attached", data=np.array(self.attached, dtype=bool))
            g = f.create_group("obs")
            for k, v in self.obs.items():
                arr = np.array(v)
                if arr.shape[1] == 1:
                    arr = arr[:, 0]
                g.create_dataset(k, data=arr)
            g.create_dataset(f"{self.camera}_rgb", data=np.array(self.rgb, dtype=np.uint8),
                             compression="gzip", compression_opts=4, chunks=(1, self.img_size, self.img_size, 3))
            g.create_dataset(f"{self.camera}_depth", data=np.array(self.depth, dtype=np.float32),
                             compression="gzip", compression_opts=4, chunks=(1, self.img_size, self.img_size))
        return path


# ----------------------------------------------------------------------------- session
class TeleopSession:
    """Shared control loop body used by the interactive viewer AND the headless self-test."""

    def __init__(self, env, device, record=False, img_size=128, out_dir=DEMO_DIR, prefix="demo", log=print):
        self.env, self.device, self.record = env, device, record
        self.out_dir, self.prefix, self.log = out_dir, prefix, log
        self.recorder = EpisodeRecorder(env, img_size) if record else None
        self.quit = False
        self.saved = []
        self.reset()

    def reset(self):
        self.obs = self.env.reset()
        self.device.start_control()
        self.device.release_all()
        self.success = False
        if self.recorder is not None:
            self.recorder.clear()
        self.log(f"[teleop] new episode | active: {ARM_LABEL[self.device.active_robot]}"
                 + (" | recording" if self.record else ""))

    def status(self):
        d = self.device
        g = ["closed" if d.gripper_closed(i) else "open" for i in range(2)]
        n = len(self.recorder) if self.recorder is not None else 0
        return (f"active: {ARM_LABEL[d.active_robot]} | gripper R:{g[0]} L:{g[1]} | success={self.success}"
                + (f" | {n} steps recorded" if self.record else ""))

    def handle_event(self, ev):
        d = self.device
        if ev == "gripper":
            d.grasp_states[d.active_robot][d.active_arm_index] = not d.grasp
        elif ev in ("switch", "arm0", "arm1"):
            d.active_robot = (1 - d.active_robot) if ev == "switch" else int(ev[-1])
        elif ev == "success":
            self.success = not self.success
        elif ev == "save":
            if self.record and len(self.recorder) > 0:
                path = self.recorder.save(self.success, self.out_dir, self.prefix)
                self.saved.append(path)
                self.log(f"[teleop] SAVED {len(self.recorder)} steps (success={self.success}) -> {path}")
            elif self.record:
                self.log("[teleop] nothing recorded yet - not saved")
            self.reset()
            return
        elif ev == "discard":
            if self.record:
                self.log(f"[teleop] discarded episode ({len(self.recorder)} steps)")
            self.reset()
            return
        elif ev == "quit":
            self.quit = True
            return
        self.log("[teleop] " + self.status())

    def step(self):
        while True:
            try:
                ev = self.device.events.get_nowait()
            except queue.Empty:
                break
            self.handle_event(ev)
            if self.quit:
                return
        action = self.device.build_action()
        if self.recorder is not None:
            self.recorder.add(self.obs, action, self.device.active_robot)
        self.obs, r, _, _ = self.env.step(action)
        if self.recorder is not None:
            self.recorder.add_reward(r)
        return action


# ----------------------------------------------------------------------------- keyboard backends
def _foreground_title():
    if sys.platform != "win32":
        return ""
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    buf = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, buf, 512)
    return buf.value


def start_pynput(device, focus_substr="MuJoCo"):
    from pynput import keyboard

    def name_of(key):
        if isinstance(key, keyboard.Key):
            return key.name
        ch = getattr(key, "char", None)
        if ch is None:
            # with modifiers held some keys only carry a virtual-key code
            vk = getattr(key, "vk", None)
            if vk is not None and (0x30 <= vk <= 0x39 or 0x41 <= vk <= 0x5A):
                return chr(vk).lower()
            return None
        return ch.lower()

    def focused():
        return focus_substr is None or focus_substr.lower() in _foreground_title().lower()

    def on_press(key):
        n = name_of(key)
        if n is not None and focused():
            device.press(n)

    def on_release(key):
        n = name_of(key)
        if n is not None:
            device.release(n)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener


# GLFW key codes used by the viewer key_callback fallback (press / repeat events only)
GLFW_NAMES = {265: "up", 264: "down", 263: "left", 262: "right", 257: "enter", 335: "enter",
              266: "page_up", 267: "page_down", 268: "home", 260: "insert",
              261: "delete", 269: "end", 256: "esc"}


def viewer_key_callback(device):
    def cb(keycode):
        n = GLFW_NAMES.get(keycode)
        if n is None and 32 <= keycode < 127:
            n = chr(keycode).lower()
        if n is None:
            return
        if n in EVENT_KEYS:
            device.press(n)
            device.release(n)
        else:
            device.press(n, hold_for=0.12)  # no release events -> emulate holding via key repeat
    return cb


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", action="store_true", help="save episodes to data/demos/*.hdf5")
    ap.add_argument("--keys", choices=["pynput", "viewer"], default="pynput",
                    help="pynput (held keys, default) or viewer (MuJoCo key_callback fallback, no key release)")
    ap.add_argument("--no-focus-check", action="store_true", help="accept keys even if the viewer is not focused")
    ap.add_argument("--pos-speed", type=float, default=0.3, help="normalised translation command when a key is held")
    ap.add_argument("--rot-speed", type=float, default=0.3, help="normalised rotation command when a key is held")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--placement-noise", type=float, default=0.0)
    ap.add_argument("--max-seconds", type=float, default=0, help="auto-quit after N s (testing)")
    args = ap.parse_args()

    import mujoco
    import mujoco.viewer

    env = make_env(has_offscreen_renderer=args.record, hard_reset=False,
                   placement_noise=args.placement_noise)
    device = TeleopKeyboard(env, args.pos_speed, args.rot_speed)
    session = TeleopSession(env, device, record=args.record, img_size=args.img_size)
    print(KEY_HELP)

    listener = None
    key_cb = None
    if args.keys == "pynput":
        try:
            listener = start_pynput(device, None if args.no_focus_check else "MuJoCo")
        except Exception as e:  # pragma: no cover
            print(f"[teleop] pynput unavailable ({e}); falling back to viewer key callback")
    if listener is None:
        key_cb = viewer_key_callback(device)

    m, d = env.sim.model._model, env.sim.data._data
    dt = 1.0 / env.control_freq
    t_start = time.perf_counter()
    last_status = 0.0
    with mujoco.viewer.launch_passive(m, d, key_callback=key_cb, show_left_ui=False,
                                     show_right_ui=False) as viewer:
        with viewer.lock():
            viewer.opt.geomgroup[0] = 0  # hide collision geoms
            viewer.cam.lookat[:] = (0.05, 0.0, 0.85)
            # behind the robots looking towards the user: screen-left = +y, screen-up/forward = +x
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.2, 0.0, -55.0
        opt_flags = viewer.opt.flags.copy()
        opt_geom, opt_site = viewer.opt.geomgroup.copy(), viewer.opt.sitegroup.copy()
        while viewer.is_running() and not session.quit:
            t0 = time.perf_counter()
            session.step()
            with viewer.lock():
                # undo the viewer's own letter shortcuts (visualisation toggles) and mark the active arm
                viewer.opt.flags[:] = opt_flags
                viewer.opt.geomgroup[:] = opt_geom
                viewer.opt.sitegroup[:] = opt_site
                scn = viewer.user_scn
                pos = session.obs[f"robot{device.active_robot}_eef_pos"]
                rgba = np.array([0.1, 0.9, 0.1, 0.25]) if device.active_robot == 0 else np.array([0.1, 0.4, 1.0, 0.25])
                mujoco.mjv_initGeom(scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.06, 0, 0]),
                                    np.asarray(pos, float), np.eye(3).ravel(), rgba.astype(np.float32))
                scn.ngeom = 1
            viewer.sync()
            now = time.perf_counter()
            if now - last_status > 2.0:
                print("[teleop] " + session.status(), flush=True)
                last_status = now
            if args.max_seconds and now - t_start > args.max_seconds:
                break
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
    if session.record and len(session.recorder) > 0:
        print(f"[teleop] quit: unsaved episode with {len(session.recorder)} steps discarded")
    if listener is not None:
        listener.stop()
    env.close()
    print(f"[teleop] done; saved {len(session.saved)} episode(s)")
    for p in session.saved:
        print("   ", p)


if __name__ == "__main__":
    main()
