from typing import Any, Dict, List, Union
import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.robots import Panda, PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

@register_env("PickCubeOcclusion-v1", max_episode_steps=150)
class PickCubeOcclusionEnv(BaseEnv):
    """
    **Task Description:**
    - Pick up a red cube and move it to a random goal position.

    **Randomizations:**
    - The cube's xy position is randomized on the table, placed flat.
    - A specified number of occluder objects from YCB are positioned above the cube.
    - The goal position is sampled on the table within a defined region.

    **Success Conditions:**
    - The cube is within `goal_thresh` (0.02m) Euclidean distance of the goal.
    - The robot is static (joint velocity < 0.2).
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]

    cube_half_size = 0.02
    goal_thresh = 0.02

    def __init__(self,
                 *args,
                 num_objects: int = 10,
                 num_occluders: int = 1,
                 robot_uids: str = "panda_wristcam",
                 robot_init_qpos_noise: float = 0.02,
                 num_envs: int = 1,
                 reconfiguration_freq: int = None,
                 **kwargs):
        # core parameters
        self.num_objects = num_objects
        self.num_occluders = num_occluders
        self.robot_init_qpos_noise = robot_init_qpos_noise

        # load YCB info and filter for occluders/distractors
        info = load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json")
        # use all keys as pool, include cube's id if present
        all_ids = np.array(list(info.keys()))
        self.occluder_model_ids = np.array(
            [
                k
                for k in load_json(
                    ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"
                ).keys()
                if k
                in [
                    "008_pudding_box",
                    "009_gelatin_box",
                    "026_sponge",
                    "030_fork",
                    "031_spoon",
                    "032_knife",
                    "033_spatula",
                    "037_scissors",
                    "042_adjustable_wrench",
                    "043_phillips_screwdriver",
                    "044_flat_screwdriver",
                    # "048_hammer",
                    "051_large_clamp",
                    "052_extra_large_clamp",
                ]
                # NOTE (arth): use these graspable objects
            ]
        )

        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0

        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs)

    @property
    def _default_sensor_configs(self):
        # front camera
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi/2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        # overhead human‑view camera
        pose = sapien_utils.look_at([0.6,0.7,0.6],[0.0,0.0,0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1.0, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # 1) build table
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        # 2) cube actor
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0,0,self.cube_half_size])
        )

        # 3) sample occluders
        num_occ = self.num_occluders
        occ_ids = self._episode_rng.choice(self.occluder_model_ids, size=num_occ, replace=False)
        choice = occ_ids

        # 3) sample occluders & distractors
        # num_occ = min(self.num_occluders, self.num_objects-1)
        # occ_ids = self._episode_rng.choice(self.occluder_model_ids, size=num_occ, replace=False)
        # choice = occ_ids
        # other_n = self.num_objects-1-num_occ
        # if other_n > 0:
        #     pool = np.setdiff1d(self.all_model_ids, occ_ids)
        #     other_ids = self._episode_rng.choice(pool, size=other_n, replace=False)
        #     choice = np.concatenate([occ_ids, other_ids])
        # else:
        #     choice = occ_ids

        self.distractors: List = []
        for i, mid in enumerate(choice):
            b = actors.get_actor_builder(self.scene, id=f"ycb:{mid}", add_collision=True, add_visual=True)
            b.set_initial_pose(sapien.Pose())
            b.set_scene_idxs(list(range(self.num_envs)))
            obj = b.build(name=f"distr_{mid}_{i}")
            self.distractors.append(obj)

        # 4) goal marker
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0,1,0,1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose()
        )
        self._hidden_objects.append(self.goal_site)

    def _after_reconfigure(self, options: dict):
        # record bottom and top offsets for cube and each distractor
        bottoms, tops = [], []
        # cube bottom
        mesh = self.cube.get_first_collision_mesh()
        bottoms.append(-mesh.bounding_box.bounds[0,2]); tops.append(mesh.bounding_box.bounds[1,2])
        # distractors
        for obj in self.distractors:
            mesh = obj.get_first_collision_mesh()
            bottoms.append(-mesh.bounding_box.bounds[0,2]); tops.append(mesh.bounding_box.bounds[1,2])
        self.bottom_offsets = common.to_tensor(bottoms, device=self.device)
        self.top_offsets = common.to_tensor(tops, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # 1) place cube flat at random XY
            cube_xy = torch.rand((b, 2), device=self.device) * 0.2 - 0.1
            cube_z  = torch.ones((b,), device=self.device) * self.cube_half_size
            cube_qs = random_quaternions(b, lock_x=True, lock_y=True)
            cube_p  = torch.cat([cube_xy, cube_z.unsqueeze(-1)], dim=1)
            self.cube.set_pose(Pose.create_from_pq(p=cube_p, q=cube_qs))

            # 2) place occluders above cube
            # for i, obj in enumerate(self.distractors, start=1):
            #     if i <= self.num_occluders:
            #         # occluder: small XY noise + random height
            #         noise = torch.rand((b,2), device=self.device)*0.04 - 0.02
            #         xy = cube_xy + noise
            #         h  = torch.rand((b,), device=self.device)*0.1 + 0.05
            #         z  = cube_z + h
            #     else:
            #         # other distractors: far from cube
            #         xy = torch.rand((b,2), device=self.device)*0.6 - 0.3
            #         d  = torch.norm(xy - cube_xy, dim=1)
            #         bad = d < 0.2
            #         while bad.any():
            #             idx = bad.nonzero(as_tuple=False).squeeze(-1)
            #             xy[idx] = torch.rand((idx.numel(),2), device=self.device)*0.6 - 0.3
            #             d = torch.norm(xy - cube_xy, dim=1); bad = d < 0.2
            #         z  = self.bottom_offsets[i].expand(b)
            #     ori = torch.tensor([1.0,0,0,0], device=self.device).repeat(b,1)
            #     pq  = torch.cat([xy, z.unsqueeze(-1), ori], dim=1)
            #     obj.set_pose(Pose.create(pq))

            # 2) place occluders above cube
            for obj in self.distractors:
                # each occluder: small XY noise + random height above cube
                noise = torch.rand((b,2), device=self.device) * 0.02 - 0.01
                xy = cube_xy + noise
                h  = torch.rand((b,), device=self.device) * 0.05 + 0.05
                z  = cube_z + h
                # ori = torch.tensor([1.0, 0, 0, 0], device=self.device).repeat(b, 1)
                ori = random_quaternions(b, lock_x=True, lock_y=True).to(self.device)
                pq  = torch.cat([xy, z.unsqueeze(-1), ori], dim=1)
                obj.set_pose(Pose.create(pq))

            # 3) sample goal on table
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            goal_xyz[:, 2] = torch.rand((b)) * 0.2 + 0.1 + cube_p[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            # 4) settle physics to avoid initial penetration
            for _ in range(50):
                self.scene.step()

            # 5) initialize robot to home configuration with small noise
            q0 = np.array([0.0, 0, 0, -2*np.pi/3, 0, 2*np.pi/3, np.pi/4, 0.04, 0.04])
            q0[:-2] += self._episode_rng.normal(0, self.robot_init_qpos_noise, len(q0)-2)
            self.agent.reset(q0)
            self.agent.robot.set_root_pose(sapien.Pose(p=[-0.615, 0, 0]))

    def _get_obs_extra(self, info: Dict) -> Dict[str, Any]:
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        return obs

    def evaluate(self) -> Dict[str, Any]:
        placed = torch.linalg.norm(self.cube.pose.p - self.goal_site.pose.p, dim=1) <= self.goal_thresh
        grasped = self.agent.is_grasping(self.cube)
        static  = self.agent.is_static(0.2)
        return dict(
            is_grasped=grasped,
            is_placed=placed,
            is_robot_static=static,
            success=placed & grasped & static,
        )