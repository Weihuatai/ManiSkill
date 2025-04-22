from typing import Any, Dict, Union

import numpy as np
import torch
import sapien
from transforms3d.euler import euler2quat, quat2euler

from mani_skill.agents.robots import Panda, PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array

@register_env("RotateCube-v1", max_episode_steps=50)
class RotateCubeEnv(BaseEnv):
    """
    **Task Description:**
    The goal is to rotate a single green cube by a specified target angle around its z-axis, using a Panda arm (optionally with a wrist camera).

    **Randomizations:**
    - The cube's XY position on the table is randomized within the region [-0.1, 0.1] to [-0.1, 0.1].
    - The cube's initial orientation is set to the identity quaternion (zero rotation).

    **Success Conditions:**
    - The cube's rotation angle, measured around its z-axis relative to its initial orientation, is within the target angle ± tolerance (30° ± 2°).
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]

    # cube parameters
    cube_half_size = 0.02
    target_angle = 30       # target rotation angle in degrees
    angle_tolerance = 2     # acceptable deviation in degrees

    def __init__(self, *args, robot_uids="panda_wristcam", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        self._initial_quat = None  # stores the cube's initial orientation

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        # build a table scene
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()

        # create a dynamic green cube at the center
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([0.1, 0.8, 0.1, 1.0]),
            name="cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size])
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # initialize table and cube pose
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # randomize cube XY position on the table
            cube_pos = torch.zeros((b, 3), device=self.device)
            cube_pos[..., :2] = torch.rand((b, 2), device=self.device) * 0.2 - 0.1
            cube_pos[..., 2] = self.cube_half_size

            # set initial orientation to identity quaternion
            self._initial_quat = torch.tensor([1, 0, 0, 0], device=self.device).repeat(b, 1)

            # apply position and orientation
            self.cube.set_pose(Pose.create_from_pq(p=cube_pos, q=self._initial_quat))

    def _get_obs_extra(self, info: Dict) -> Dict:
        # always provide TCP pose
        obs = {"tcp_pose": self.agent.tcp.pose.raw_pose}
        # include state if requested
        if self.obs_mode_struct.use_state:
            obs.update(
                cube_pose=self.cube.pose.raw_pose,
                initial_quat=self._initial_quat
            )
        return obs

    def evaluate(self) -> Dict[str, torch.Tensor]:
        # compute current rotation delta from initial orientation
        current_quat = self.cube.pose.q
        angle_diff = self.quaternion_angle_diff(current_quat, self._initial_quat)

        # success if within target ± tolerance
        success = (
            angle_diff >= (self.target_angle - self.angle_tolerance)
        ) & (
            angle_diff <= (self.target_angle + self.angle_tolerance)
        )
        return {"success": success, "angle_diff": angle_diff}

    @staticmethod
    def quaternion_angle_diff(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Compute rotation angle difference (in degrees) between two quaternions.
        """
        dot = torch.sum(q1 * q2, dim=1)
        angle_rad = 2 * torch.acos(torch.clamp(torch.abs(dot), 0.0, 1.0))
        return torch.rad2deg(angle_rad)

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict) -> torch.Tensor:
        # sparse reward: +1 on success, 0 otherwise
        success = self.evaluate()["success"]
        return success.to(torch.float32)

    def compute_normalized_dense_reward(
        self, obs: Any, action: Array, info: Dict
    ) -> torch.Tensor:
        # normalized dense reward same as sparse
        return self.compute_dense_reward(obs, action, info)