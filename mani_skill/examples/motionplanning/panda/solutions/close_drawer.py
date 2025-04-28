import gymnasium as gym
import numpy as np
import sapien

from mani_skill.envs.tasks import CloseDrawerEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.utils import common
from transforms3d.euler import euler2quat

def main():
    env: CloseDrawerEnv = gym.make(
        "CloseDrawer-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
    )

def solve(env: CloseDrawerEnv, seed=None, debug=False, vis=False):
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

    # TODO: some drawer is not open, so need to pass
    current_qpos = env.handle_link.joint.qpos
    if current_qpos < 0.1:
        return -1

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

    # Set the gripper pose
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

    push_dist = float(current_qpos)
    # motion parameters
    push_dir = -approaching
    current_pose = grasp_pose
    z_fixed = grasp_pose.p[2]
    step = 0.1
    total = 0.0

    while total < push_dist:
        delta = min(step, push_dist - total)
        next_pose = current_pose * sapien.Pose(p=delta * push_dir)
        p = next_pose.p
        # Keep constant height
        p[2] = z_fixed
        next_pose = sapien.Pose(p, next_pose.q)
        res = planner.move_to_pose_with_screw(next_pose, dry_run=False)
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

if __name__ == "__main__":
    main()
