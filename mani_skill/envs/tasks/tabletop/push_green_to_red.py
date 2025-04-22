from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Panda, PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig

# TODO: Add reward function

@register_env("PushGreenToRed-v1", max_episode_steps=50)
class PushGreenToRedEnv(BaseEnv):
    """
    Task Description:
    Push the green cube to make contact with the red cube on a tabletop environment
    
    Randomizations:
    - Green cube' xy positions are randomized on the table within [-0.10, -0.10] to [0.10, 0.10]
    - The robot's initial joint configuration is perturbed by Gaussian noise (std=0.02)
    - The red cube is kinematic
    - The red cube is placed at a fixed offset along the x-axis relative to the green cubestarting position (0.15 m away)
    
    Success Conditions:
    - The distance between the centers of the two cubes in the x-y plane becomes less than or equal to the cube's full width (0.04 m) plus a margin of 0.01 m.
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wrist_cam"]
    agent: Union[Panda, PandaWristCam]
    
    # cube parameters
    cube_half_size = 0.02
    # distance between cubes
    min_distance = 0.15

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, 
                max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi/2,
                near=0.01,
                far=100
            )
        ]

    @property 
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # Build table scene
        self.table_scene = TableSceneBuilder(
            env=self, 
            robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Green cube (movable)
        self.green_cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([0, 1, 0, 1]),  # Green
            name="green_cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size])
        )

        # Red cube (target, kinematic)
        self.red_cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([1, 0, 0, 1]),  # Red
            name="red_cube",
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size])
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Sample random x-y for green cube within the specified square
            pos_green = torch.zeros((b, 3), device=self.device)
            for i in range(b):
                p = torch.rand(2, device=self.device) * 0.2 - 0.1  # [-0.1, 0.1]
            pos_green[i, :2] = p
            pos_green[..., 2] = self.cube_half_size

            # Place red cube at fixed offset along x-axis
            pos_red = pos_green.clone()
            pos_red[..., 0] += self.min_distance
            self.green_cube.set_pose(Pose.create_from_pq(p=pos_green))
            self.red_cube.set_pose(Pose.create_from_pq(p=pos_red))

    def evaluate(self):
        # Calculate distance between cube centers
        dist = torch.linalg.norm(
            self.green_cube.pose.p[:, :2] - self.red_cube.pose.p[:, :2], 
            dim=1
        )
        
        # Check if cubes are in contact (allow small margin)
        success = dist <= (2 * self.cube_half_size) + 0.01
        return {
            "success": success,
            "cube_distance": dist
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                red_cube_pose=self.red_cube.pose.raw_pose,
                green_cube_pose=self.green_cube.pose.raw_pose
            )
        return obs

    # Remove dense reward functions to use default sparse reward
    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        return self.compute_dense_reward(obs, action, info)