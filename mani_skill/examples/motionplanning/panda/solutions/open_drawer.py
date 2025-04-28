import gymnasium as gym
import numpy as np
import sapien

from mani_skill.envs.tasks import OpenDrawerEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.utils import common
from transforms3d.euler import euler2quat

def main():
    env: OpenDrawerEnv = gym.make(
        "OpenDrawer-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
    )

def solve(env: OpenDrawerEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in ["pd_joint_pos", "pd_joint_pos_vel"]
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=0.75,
        joint_acc_limits=0.75
    )
    env = env.unwrapped
    FINGER_LENGTH = 0.025

    # Use the center of handle_link_goal as the grasp target
    center = env.handle_link_goal.pose.p.numpy()  # [3]
    approaching = np.array([0, 0, -1]) # Approach direction along the negative z-axis
    # Use TCP's y-axis as the initial gripper closing direction
    tcpmat = env.agent.tcp.pose.to_transformation_matrix()
    target_closing = tcpmat[0, :3, 1].numpy()

    # Orthogonalize the closing direction relative to approaching direction
    closing = target_closing - (approaching @ target_closing) * approaching
    closing = common.np_normalize_vector(closing)
    handle_init_pose = env.agent.build_grasp_pose(approaching, closing, center)

    rot_q = euler2quat(np.deg2rad(75), 0, np.pi/2) 

    # Move to initial pose
    initial_pose = handle_init_pose * sapien.Pose(q=rot_q) * (sapien.Pose([0, 0, -0.05]))
    res = planner.move_to_pose_with_screw(initial_pose, dry_run=False, refine_steps=20)
    if res == -1: return -1
    
    # Move to grasp pose
    grasp_pose = initial_pose * sapien.Pose([0, 0.02, 0.05])
    res = planner.move_to_pose_with_screw(grasp_pose, dry_run=False, refine_steps=20)
    if res == -1: return -1
    planner.close_gripper()

    # Pull action configuration
    pull_dir = approaching
    pull_dist = env.target_qpos.item() * 1.25 # 0.80*1.25 = 1
    step = 0.1
    total = 0.0
    current_pose = grasp_pose
    z_fixed = grasp_pose.p[2]  # Keep the initial z-axis height during pulling


    while total < pull_dist:
        delta = min(step, pull_dist - total)
        next_pose = current_pose * sapien.Pose(p=delta * pull_dir)
        # Lock the z-axis height to prevent drifting
        p = next_pose.p
        p[2] = z_fixed
        next_pose = sapien.Pose(p, next_pose.q)
        res = planner.move_to_pose_with_screw(next_pose, dry_run=False, refine_steps=50)
        if res == -1:
            return -1
        current_pose = next_pose
        total += delta

    # Direct one-time pull in place
    # next_pose = current_pose * sapien.Pose(p=pull_dist * pull_dir)
    # p = next_pose.p
    # p[2] = z_fixed
    # next_pose = sapien.Pose(p, next_pose.q)
    # res = planner.move_to_pose_with_screw(next_pose, dry_run=False)
    # if res == -1:
    #     return -1

    return res