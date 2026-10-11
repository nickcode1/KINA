"""BimanualAssist: two Franka Panda arms side by side at a table, set up for
feeding (fork, plate, food, user's mouth) and jar opening (jar with hinged lid)."""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor

from .objects import ForkObject, JarObject, make_food, make_plate

N_FOOD = 3


def look_at_quat(pos, target, up=(0, 0, 1)):
    """MuJoCo camera quaternion (w,x,y,z) for a camera at `pos` looking at `target`."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    mat = np.stack([x, y, -f], axis=1)  # camera looks along its -z axis
    q = T.mat2quat(mat)  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])


def _fmt(v):
    return " ".join(f"{x:.4f}" for x in v)


class BimanualAssist(TwoArmEnv):
    """
    Robot 0 (y < 0) is the RIGHT arm (fork / feeding, lid twisting), robot 1 (y > 0) is the
    LEFT arm (holds the jar). The user's mouth is a fixed visual marker across the table.

    World frame: table top centre at (0, 0, 0.8); robots on the -x side, user on the +x side.
    """

    # --- scene layout (world frame, metres) ---
    TABLE_SIZE = (0.45, 1.0, 0.05)        # small table so the far side (user) is reachable
    TABLE_OFFSET = (0.0, 0.0, 0.8)
    ARM_Y_OFFSETS = (-0.22, 0.22)          # robot0 = right arm, robot1 = left arm
    FORK_POS = (-0.07, -0.33)              # handle centre; fork points along +x
    PLATE_POS = (0.04, -0.09)
    FOOD_OFFSETS = ((-0.03, 0.0), (0.03, 0.025), (0.005, -0.035))  # relative to plate centre
    JAR_POS = (0.0, 0.25)
    MOUTH_POS = (0.36, -0.05, 1.15)        # seated user's mouth, ~35 cm above the table
    RGBD_CAM_POS = (-0.95, 0.0, 2.2)
    RGBD_CAM_TARGET = (0.12, 0.0, 0.8)
    LID_TURNS_TO_OPEN = 2.0

    def __init__(
        self,
        robots,
        env_configuration="parallel",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        use_camera_obs=False,
        use_object_obs=True,
        placement_noise=0.0,
        lid_turns_to_open=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=True,
        hard_reset=True,
        camera_names="rgbd_cam",
        camera_heights=480,
        camera_widths=640,
        camera_depths=True,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        self.use_object_obs = use_object_obs
        self.placement_noise = placement_noise
        if lid_turns_to_open is not None:
            self.LID_TURNS_TO_OPEN = float(lid_turns_to_open)  # instance override of the class default
        self.table_offset = np.array(self.TABLE_OFFSET)
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    # ------------------------------------------------------------------ model
    def _load_model(self):
        super()._load_model()
        assert self.env_configuration == "parallel", "BimanualAssist expects two arms in 'parallel'"
        for robot, dy in zip(self.robots, self.ARM_Y_OFFSETS):
            xpos = np.array(robot.robot_model.base_xpos_offset["table"](self.TABLE_SIZE[0]))
            robot.robot_model.set_base_xpos(xpos + np.array((0, dy, 0)))

        arena = TableArena(table_full_size=self.TABLE_SIZE, table_friction=(1.0, 5e-3, 1e-4),
                           table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        # RGB-D camera behind and above the robots, looking at the table
        arena.set_camera("rgbd_cam", self.RGBD_CAM_POS,
                         look_at_quat(self.RGBD_CAM_POS, self.RGBD_CAM_TARGET), {"fovy": "55"})
        # Overview camera from the user's side, above
        ov_pos, ov_tgt = (1.3, -1.1, 1.9), (0.0, 0.0, 0.9)
        arena.set_camera("overview", ov_pos, look_at_quat(ov_pos, ov_tgt), {"fovy": "50"})
        # Mouth: visual-only marker (no collision) + translucent head for context
        head = ET.SubElement(arena.worldbody, "body", name="user_head",
                             pos=_fmt(np.add(self.MOUTH_POS, (0.08, 0, 0.05))))
        ET.SubElement(head, "geom", name="user_head_vis", type="sphere", size="0.09",
                      rgba="0.95 0.8 0.65 0.35", contype="0", conaffinity="0", group="1")
        mouth = ET.SubElement(arena.worldbody, "body", name="mouth", pos=_fmt(self.MOUTH_POS))
        ET.SubElement(mouth, "geom", name="mouth_vis", type="sphere", size="0.015",
                      rgba="1 0 0 1", contype="0", conaffinity="0", group="1")
        ET.SubElement(mouth, "site", name="mouth_site", type="sphere", size="0.005", rgba="1 0 0 0")

        self.fork = ForkObject("fork")
        self.jar = JarObject("jar")
        self.plate = make_plate("plate")
        self.foods = [make_food(i) for i in range(N_FOOD)]

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.fork, self.plate, *self.foods, self.jar],
        )
        # physics at 500 Hz (control at self.control_freq = 20 Hz)
        option = self.model.root.find("option")
        option.set("timestep", "0.002")
        # multi-point convex contacts: keeps small cylinders/boxes from jittering on the plate
        flag = option.find("flag") if option.find("flag") is not None else ET.SubElement(option, "flag")
        flag.set("multiccd", "enable")
        # fork-food weld placeholders (inactive); see attach_food()/detach_food()
        for i, food in enumerate(self.foods):
            ET.SubElement(self.model.equality, "weld", name=f"fork_food{i}_weld",
                          body1=self.fork.root_body, body2=food.root_body, active="false",
                          solref="0.01 1")

    def _setup_references(self):
        super()._setup_references()
        m = self.sim.model
        self.fork_body_id = m.body_name2id(self.fork.root_body)
        self.fork_tip_sid = m.site_name2id(self.fork.important_sites["tip"])
        self.food_body_ids = [m.body_name2id(f.root_body) for f in self.foods]
        self.plate_body_id = m.body_name2id(self.plate.root_body)
        self.jar_body_id = m.body_name2id(self.jar.root_body)
        self.mouth_sid = m.site_name2id("mouth_site")
        self.lid_qpos_addr = m.get_joint_qpos_addr(self.jar.lid_joint)
        self.weld_ids = [mujoco.mj_name2id(m._model, mujoco.mjtObj.mjOBJ_EQUALITY, f"fork_food{i}_weld")
                         for i in range(N_FOOD)]

    def _setup_observables(self):
        observables = super()._setup_observables()
        if not self.use_object_obs:
            return observables

        def d():
            return self.sim.data  # sim may be re-created on hard reset

        sensors = {
            "fork_tip_pos": lambda c: np.array(d().site_xpos[self.fork_tip_sid]),
            "fork_pos": lambda c: np.array(d().body_xpos[self.fork_body_id]),
            "fork_quat": lambda c: T.convert_quat(np.array(d().body_xquat[self.fork_body_id]), to="xyzw"),
            "plate_pos": lambda c: np.array(d().body_xpos[self.plate_body_id]),
            "jar_pos": lambda c: np.array(d().body_xpos[self.jar_body_id]),
            "jar_quat": lambda c: T.convert_quat(np.array(d().body_xquat[self.jar_body_id]), to="xyzw"),
            "lid_angle": lambda c: np.array([d().qpos[self.lid_qpos_addr]]),
            "lid_open": lambda c: np.array([float(self.is_jar_open())]),
            "mouth_pos": lambda c: np.array(d().site_xpos[self.mouth_sid]),
            "fork_tip_to_mouth": lambda c: np.array(d().site_xpos[self.mouth_sid] - d().site_xpos[self.fork_tip_sid]),
        }
        for i in range(N_FOOD):
            sensors[f"food{i}_pos"] = (lambda k: (lambda c: np.array(d().body_xpos[self.food_body_ids[k]])))(i)
        for name, fn in sensors.items():
            fn.__name__ = name
            observables[name] = Observable(name=name, sensor=sensor(modality="object")(fn),
                                           sampling_rate=self.control_freq)
        return observables

    def _reset_internal(self):
        super()._reset_internal()
        if self.deterministic_reset:
            return
        z0 = self.table_offset[2]
        rng = getattr(self, "rng", None) or np.random.default_rng()

        def noise():
            return rng.uniform(-1, 1, 2) * self.placement_noise

        def place(obj, xy, z):
            self.sim.data.set_joint_qpos(obj.naming_prefix + "joint0", np.r_[xy, z, 1, 0, 0, 0])

        place(self.fork, np.add(self.FORK_POS, noise()), z0 + ForkObject.REST_HALF_HEIGHT + 0.0005)
        plate_xy = np.add(self.PLATE_POS, noise())
        place(self.plate, plate_xy, z0 + 0.006 + 0.0005)
        for food, off in zip(self.foods, self.FOOD_OFFSETS):
            half_h = food.size[-1]  # box: half-size z; cylinder: half-length
            place(food, plate_xy + np.array(off), z0 + 0.012 + half_h + 0.001)
        place(self.jar, np.add(self.JAR_POS, noise()), z0 + JarObject.BOTTOM_OFFSET + 0.0005)
        self.sim.data.qpos[self.lid_qpos_addr] = 0.0
        self.sim.forward()

    # ------------------------------------------------------------------ task
    def reward(self, action=None):
        return 0.0  # TODO: task rewards (feeding / jar opening) in later weeks

    def _check_success(self):
        return False

    # ------------------------------------------------------------------ helpers
    @property
    def lid_angle(self):
        """Lid rotation about the jar axis [rad] (sign: +z right-hand rule)."""
        return float(self.sim.data.qpos[self.lid_qpos_addr])

    @property
    def lid_turns(self):
        """Lid rotation in full turns (signed)."""
        return self.lid_angle / (2 * np.pi)

    def is_jar_open(self):
        """True once the lid has been turned >= LID_TURNS_TO_OPEN full turns (either direction)."""
        return abs(self.lid_angle) >= 2 * np.pi * self.LID_TURNS_TO_OPEN

    @property
    def lid_open(self):
        return self.is_jar_open()

    def attach_food(self, i):
        """Activate the fork-food weld for piece i, freezing the CURRENT relative pose."""
        m, d = self.sim.model._model, self.sim.data._data
        b1, b2 = self.fork_body_id, self.food_body_ids[i]
        q1inv = np.zeros(4)
        mujoco.mju_negQuat(q1inv, d.xquat[b1])
        relpos = np.zeros(3)
        mujoco.mju_rotVecQuat(relpos, d.xpos[b2] - d.xpos[b1], q1inv)
        relquat = np.zeros(4)
        mujoco.mju_mulQuat(relquat, q1inv, d.xquat[b2])
        eid = self.weld_ids[i]
        m.eq_data[eid, 0:3] = 0.0       # anchor (body2 frame)
        m.eq_data[eid, 3:6] = relpos    # body2 pose in body1 frame
        m.eq_data[eid, 6:10] = relquat
        m.eq_data[eid, 10] = 1.0        # torque scale
        d.eq_active[eid] = 1

    def detach_food(self, i):
        self.sim.data._data.eq_active[self.weld_ids[i]] = 0

    def food_attached(self, i):
        return bool(self.sim.data._data.eq_active[self.weld_ids[i]])


BimanualAssistEnv = BimanualAssist
