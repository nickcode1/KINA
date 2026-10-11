# Keyboard teleoperation & demo recording (Week 2)

## Quick start

```powershell
teleop.bat                                    # double-click: teleop + recording -> data\demos\*.hdf5
.venv\Scripts\python scripts\teleop.py        # teleop only, nothing saved
.venv\Scripts\python scripts\teleop.py --record --pos-speed 0.5 --rot-speed 0.3
.venv\Scripts\python scripts\replay_demo.py   # replay newest demo -> outputs\replay_<name>.mp4 (+ _first/_last.png)
.venv\Scripts\python scripts\test_teleop_headless.py   # headless self-test (synthetic keys -> record -> replay)
run_demo.bat                                  # double-click: scripted fork-grasp video, then the viewer
```

Extra dependency: `pynput` (in `requirements.txt`; `teleop.bat` installs it if missing).

**Click the MuJoCo window first.** Keys only count while the viewer window has keyboard focus
(pynput is a global hook; teleop.py filters on the foreground window title "MuJoCo").
The console prints the active arm, gripper states, success flag and number of recorded steps every 2 s
and on every change. In the viewer, a translucent sphere marks the active arm's gripper
(**green = RIGHT arm / robot0**, **blue = LEFT arm / robot1**).

## Key bindings / 按键

Directions are in the world frame: the robots sit at -x and face the user at +x. The default
view is from behind the robots, so Up arrow = away from you (towards the user's mouth), Left arrow = screen left.
Hold a movement key to keep moving (velocity control at 20 Hz); release to stop.

| Key | Alt key¹ | Action | 功能 |
|---|---|---|---|
| ↑ / ↓ | | move +x / −x (towards user / back) | 前进 / 后退 |
| ← / → | | move +y / −y (robots' left / right) | 左移 / 右移 |
| E / Q | PgUp / PgDn | move up / down (+z / −z) | 上升 / 下降 |
| Z / X | 3 / 4 | roll + / − (about x) | 绕 x 轴旋转 (roll) |
| C / V | 5 / 6 | pitch + / − (about y) | 绕 y 轴旋转 (pitch) |
| A / D | 7 / 8 | yaw + / − (about z) | 绕 z 轴旋转 (yaw) |
| hold Shift | | slow / fine mode (×0.3) | 精细慢速模式 |
| F | 9 | toggle gripper of active arm (open ↔ closed) | 开合当前手臂夹爪 |
| N | 0 | switch active arm (right ↔ left) | 切换左右手臂 |
| 1 / 2 | | select RIGHT arm (robot0) / LEFT arm (robot1) | 选择右臂 / 左臂 |
| Y | Insert | toggle episode success flag | 标记/取消本次成功 |
| Enter | Home | end episode: **save** (with `--record`) + reset | 保存本次演示并重置 |
| Delete | End | end episode: **discard** + reset | 丢弃本次演示并重置 |
| Esc / close window | | quit (unsaved episode is discarded) | 退出（未保存的丢弃） |

¹ The alt keys also work with `--keys viewer` (fallback without pynput: MuJoCo's own key callback,
which never sees letters or Enter because the viewer consumes them, and has no key-release events,
so holding a key only moves continuously after the OS key-repeat delay).

Keys to avoid in the viewer: **Space** (viewer pause), **Backspace** (viewer resets the sim data),
**Tab / Shift+Tab** (side panels), **S W R L K G / ,** (render flags: shadow, wireframe, reflection, ...).
Other letter/digit shortcuts of the viewer (contact points, equality, geom groups, ...) are undone every frame.

Speeds: a held key commands a normalised OSC delta of `--pos-speed` (default 0.3 → 1.5 cm per 50 ms step
target offset, ~7 cm/s actual) and `--rot-speed` (default 0.3 → 0.15 rad offset per step).

## How it works

- `scripts/teleop.py`
  - `TeleopKeyboard(robosuite.devices.Device)`: held-key set → 6-d velocity command; reuses robosuite's
    `Device.input2action()` (OSC delta scaling, gripper ±1, `active_robot` / `active_arm` logic) like
    `robosuite/scripts/collect_human_demonstrations.py`. Our arms are two single-arm robots
    (robot0 = RIGHT, robot1 = LEFT), so switching arms switches `active_robot`. The inactive arm
    gets a zero delta and keeps its last gripper command.
  - `TeleopSession.step()`: processes queued key events, builds the 14-d action, records, steps the env.
    The interactive loop and the headless test use the same code.
  - env: `make_env(hard_reset=False, has_offscreen_renderer=<--record>)` (hard_reset off so the
    viewer keeps valid model/data pointers across resets).
- `scripts/replay_demo.py file.hdf5 [--mode states|actions] [--camera overview]`
  - `states` (default): sets the recorded flattened MuJoCo state each step (exact).
  - `actions`: starts from the first state and re-applies the actions open-loop; prints EEF drift
    (self-test: ≤ 3 mm right arm, ≤ 8 mm left arm over 140 steps).
  - Video frame = replayed render | recorded `rgbd_cam` RGB | recorded depth (colour-mapped).
  - The model is rebuilt from the XML stored in the file (`reset_from_xml_string`).

## Data format (`data/demos/<prefix>_YYYYmmdd_HHMMSS.hdf5`, one file per episode)

T = number of control steps (20 Hz). `obs/*[t]`, `states[t]` and the images are recorded **before**
`actions[t]` is applied (standard (o_t, a_t) pairs).

| Path | Shape / type | Content |
|---|---|---|
| attrs `success` | bool | user flag (Y key) |
| attrs `num_steps`, `control_freq`, `date`, `env`, `robots`, `controller`, `action_layout`, `camera`, `image_size`, `image_convention` | | metadata |
| attrs `model_file` | str | full MJCF XML (for replay) |
| `actions` | (T, 14) float32 | `[robot0/RIGHT: dx dy dz drx dry drz grip, robot1/LEFT: same]`, normalised [-1, 1]; pos 1.0 = 0.05 m, rot 1.0 = 0.5 rad (axis-angle, base = world frame); grip −1 open / +1 close |
| `states` | (T, 117) float64 | `sim.get_state().flatten()` (time, qpos, qvel) |
| `rewards` | (T,) float32 | env reward (currently 0) |
| `active_robot` | (T,) int8 | arm being teleoperated (0 = right, 1 = left) |
| `food_attached` | (T, 3) bool | fork–food weld active flags |
| `obs/robot{0,1}_joint_pos` | (T, 7) | arm joint positions (also `_joint_vel`, `_joint_acc`, `_joint_pos_cos/sin`) |
| `obs/robot{0,1}_gripper_qpos` | (T, 2) | finger joint positions (also `_gripper_qvel`) |
| `obs/robot{0,1}_eef_pos`, `_eef_quat` | (T, 3), (T, 4) | end-effector pose (quat xyzw) |
| `obs/fork_tip_pos`, `fork_pos`, `fork_quat`, `fork_tip_to_mouth` | (T, 3/4) | fork |
| `obs/food{0,1,2}_pos`, `plate_pos`, `jar_pos`, `jar_quat`, `mouth_pos` | (T, 3/4) | objects |
| `obs/lid_angle`, `obs/lid_open` | (T,) | jar lid angle [rad], open flag |
| `obs/rgbd_cam_rgb` | (T, 128, 128, 3) uint8, gzip | RGB from `rgbd_cam`, row 0 = top |
| `obs/rgbd_cam_depth` | (T, 128, 128) float32, gzip | depth in **metres**, row 0 = top |

Size: ~45 kB per step (~0.9 MB per second of demo).

Reading for ACT-style training:

```python
import h5py, numpy as np
with h5py.File("data/demos/demo_....hdf5") as f:
    qpos = np.concatenate([f["obs/robot0_joint_pos"][:], f["obs/robot0_gripper_qpos"][:],
                           f["obs/robot1_joint_pos"][:], f["obs/robot1_gripper_qpos"][:]], 1)  # (T, 18)
    img, depth, act = f["obs/rgbd_cam_rgb"][:], f["obs/rgbd_cam_depth"][:], f["actions"][:]     # act (T, 14)
```

## Verification done

- `scripts/test_teleop_headless.py`: synthetic key presses through the same TeleopKeyboard /
  TeleopSession path; checks right +x (Up) +9.0 cm, right +z (E) +7.7 cm, left +y (Left) +8.9 cm,
  left −z (Shift+Q) +2.3 cm, idle arm moves < 1 cm, gripper −1 → +1, success flag saved; then replays.
- Real OS key events (pynput `Controller`) sent to the live viewer window: episode saved with the right
  EEF moving +11.4 cm x / +7.5 cm z and the left +11.4 cm y; discard / quit keys worked; also with `--keys viewer`.

## Notes / limitations

- pynput is a global keyboard hook; on Windows it works without admin rights. Keys typed while another
  window is focused are ignored (title filter). `--no-focus-check` disables the filter.
- Food skewering (`env.attach_food`) is not bound to a key yet; flattened states do not contain the
  weld activation (`food_attached` is stored separately for that).
- Quitting (Esc / closing the window) discards the current unsaved episode — press Enter first.
