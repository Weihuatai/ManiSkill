import os
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
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

# NOTE: Due to the way it is generated, the number of occluders cannot be well specified

@register_env(
    "PickBananaOcclusion-v1",
    max_episode_steps=150,
    asset_download_ids=["ycb"],
)
class PickBananaOcclusionEnv(BaseEnv):
    """
    **Task Description:**
    - Pick up a banana from the [YCB dataset](https://www.ycbbenchmarks.com/) and move it to a random goal position.

    **Randomizations:**
    - Banana is placed flat on the table with random XY within the table bounds.
    - A specified number of occluder objects are positioned above the banana to occlude it.
    - Additional distractor objects are scattered around the table without contacting the banana.
    - The goal position is sampled randomly on the table within a defined region.

    **Success Conditions:**
    - The banana's position is within goal_thresh (default 0.02) Euclidean distance of the goal.
    - The robot joints are static (joint velocity < 0.2).
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]
    goal_thresh = 0.02

    def __init__(
        self,
        *args,
        num_objects: int = 10,
        num_occluders: int = 3,
        robot_uids: str = "panda_wristcam",
        robot_init_qpos_noise: float = 0.02,
        num_envs: int = 1,
        reconfiguration_freq: int = None,
        **kwargs,
    ):
        self.num_objects = num_objects
        self.num_occluders = num_occluders
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.banana_id = "011_banana"
        # model ids eligible as occluders
        self.occluder_model_ids = np.array(
            [
                k
                for k in load_json(
                    ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"
                ).keys()
                if k
                not in [
                    "002_master_chef_can","003_cracker_box","004_sugar_box","005_tomato_soup_can",
                    "006_mustard_bottle","007_tuna_fish_can","010_potted_meat_can",
                    "011_banana",
                    "012_strawberry","013_apple","014_lemon","015_peach","016_pear","017_orange",
                    "018_plum",
                    "019_pitcher_base",
                    "021_bleach_cleanser",
                    "022_windex_bottle",
                    "024_bowl",
                    "025_mug",
                    "028_skillet_lid",
                    "029_plate",
                    "035_power_drill","036_wood_block",
                    "040_large_marker","050_medium_clamp",
                    "053_mini_soccer_ball","054_softball","055_baseball","056_tennis_ball","057_racquetball","058_golf_ball",
                    "059_chain",
                    "061_foam_brick",
                    "062_dice",
                    "063-a_marbles","063-b_marbles",
                    "065-a_cups","065-b_cups","065-c_cups","065-d_cups","065-e_cups","065-f_cups","065-g_cups","065-h_cups","065-i_cups","065-j_cups",
                    "070-a_colored_wood_blocks","070-b_colored_wood_blocks",
                    "072-a_toy_airplane","072-b_toy_airplane","072-c_toy_airplane","072-d_toy_airplane","072-e_toy_airplane",
                    "073-a_lego_duplo","073-b_lego_duplo","073-c_lego_duplo","073-d_lego_duplo","073-e_lego_duplo","073-f_lego_duplo","073-g_lego_duplo",
                    "077_rubiks_cube"
                ]  
                # NOTE (arth): ignore these non-graspable/hard to grasp ycb objects
                # NOTE ：also ignore banana and some objects not suitable for occluding others
            ]
        )

        # all model ids for additional distractors
        self.all_model_ids = np.array(
            [
                k
                for k in load_json(
                    ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"
                ).keys()
                if k
                not in [
                    "022_windex_bottle",
                    "028_skillet_lid",
                    "029_plate",
                    "059_chain",
                ]  # NOTE (arth): ignore these non-graspable/hard ycb objects
            ]
        )

        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2 ** 20, max_rigid_patch_count=2 ** 19
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table.build()

        ban_builder = actors.get_actor_builder(
            self.scene, id=f"ycb:{self.banana_id}", add_collision=True, add_visual=True
        )
        ban_builder.set_initial_pose(sapien.Pose())
        ban_builder.set_scene_idxs(list(range(self.num_envs)))
        self.banana = ban_builder.build(name=self.banana_id)

        # sample occluders and distractors
        # first num_occluders from occluder_model_ids
        num_occ = min(self.num_occluders, self.num_objects - 1)
        occluder_ids = self._episode_rng.choice(
            self.occluder_model_ids, size=num_occ, replace=False
        )
        # remaining from all_model_ids
        other_n = self.num_objects - 1 - num_occ
        if other_n > 0:
            other_pool = np.setdiff1d(self.all_model_ids, occluder_ids)
            other_ids = self._episode_rng.choice(
                other_pool, size=other_n, replace=False
            )
            choice = np.concatenate([occluder_ids, other_ids])
        else:
            choice = occluder_ids

        self.distractors: List[Actor] = []
        for i, mid in enumerate(choice):
            builder = actors.get_actor_builder(
                self.scene, id=f"ycb:{mid}", add_collision=True, add_visual=True
            )
            builder.set_initial_pose(sapien.Pose())
            builder.set_scene_idxs(list(range(self.num_envs)))
            obj = builder.build(name=f"{mid}_{i}")
            self.distractors.append(obj)

        self.objects = None

        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _after_reconfigure(self, options: dict):
        # record bottom and top bounding offsets for each object
        bottoms, tops = [], []
        for obj in [self.banana] + self.distractors:
            mesh = obj.get_first_collision_mesh()
            # mesh.bounding_box.bounds[:,2] = [−half_height, +half_height]
            bot = -mesh.bounding_box.bounds[0, 2]
            top =  mesh.bounding_box.bounds[1, 2]
            bottoms.append(bot)
            tops.append(top)
        self.object_bottom_offsets = common.to_tensor(bottoms, device=self.device)
        self.object_top_offsets = common.to_tensor(tops, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table.initialize(env_idx)

            # 1) place banana at random XY on table, keep flat orientation
            ban_xy = torch.rand((b, 2), device=self.device) * 0.3 - 0.15
            ban_z = self.object_bottom_offsets[0].expand(b)
            ban_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(b, 1)
            # ban_q = random_quaternions(b, lock_x=True, lock_y=True)
            ban_pq = torch.cat([ban_xy, ban_z.unsqueeze(-1), ban_q], dim=1)  # (b,7) raw_pose
            self.banana.set_pose(Pose.create(ban_pq))

            # 2) place distractors
            for i, obj in enumerate(self.distractors, start=1):
                # The first num_occluders are near bananas 
                if (i - 1) < self.num_occluders:
                    noise = torch.rand((b, 2), device=self.device) * 0.06 - 0.03
                    xy = ban_xy + noise
                    # occluders: position above banana with small XY noise and margin
                    margin = torch.rand((b,), device=self.device) * 0.02 + 0.05
                    z = ban_z + margin
                else:
                    # other distractors: keep at least 0.2m away from banana XY
                    margin = 0.2
                    xy = torch.rand((b, 2), device=self.device) * 0.6 - 0.3
                    d = torch.norm(xy - ban_xy, dim=1)
                    bad = d <= margin
                    while bad.any():
                        idx = bad.nonzero(as_tuple=False).squeeze(-1)
                        new_xy = torch.rand((idx.numel(), 2), device=self.device) * 0.6 - 0.3
                        xy[idx] = new_xy
                        d = torch.norm(xy - ban_xy, dim=1)
                        bad = d <= margin
                    z = self.object_bottom_offsets[i].expand(b)

                # ori = random_quaternions(b, lock_x=True, lock_y=True)
                ori = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(b, 1)
                pq = torch.cat([xy, z.unsqueeze(-1), ori], dim=1)
                obj.set_pose(Pose.create(pq))

            # 3) sample goal position above table near banana
            goal_xy = torch.rand((b, 2), device=self.device) * 0.3 - 0.15
            goal_z = torch.rand((b,), device=self.device) * 0.2 + 0.3  + ban_z
            goal_p = torch.stack([goal_xy[:, 0], goal_xy[:, 1], goal_z], dim=1)
            self.goal_site.set_pose(Pose.create_from_pq(p=goal_p))
            
            # 4) step physics to settle objects
            for _ in range(50):
                self.scene.step()

            # 5) initialize robot joint configuration and base pose
            q0 = np.array([0.0, 0, 0, -2*np.pi/3, 0, 2*np.pi/3, np.pi/4, 0.04, 0.04])
            q0[:-2] += self._episode_rng.normal(0, self.robot_init_qpos_noise, len(q0)-2)
            self.agent.reset(q0)
            self.agent.robot.set_root_pose(sapien.Pose([-0.615, 0, 0]))

    def evaluate(self):
        placed = torch.linalg.norm(self.banana.pose.p - self.goal_site.pose.p, dim=1) < self.goal_thresh
        grasped = self.agent.is_grasping(self.banana)
        static = self.agent.is_static(0.2)
        return dict(
            is_grasped=grasped,
            is_placed=placed,
            is_robot_static=static,
            success=placed & grasped & static,
        )

    def _get_obs_extra(self, info: Dict) -> Dict[str, Any]:
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            banana_pos=self.banana.pose.p,
            goal_pos=self.goal_site.pose.p,
            is_grasped=info["is_grasped"],
        )
        if "state" in self.obs_mode:
            obs.update(
                tcp_to_obj=self.banana.pose.p - self.agent.tcp.pose.p,
                obj_to_goal=self.goal_site.pose.p - self.banana.pose.p,
            )
        return obs