import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.pick_banana_margin import PickBananaMarginEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import get_actor_obb, compute_grasp_info_by_obb


def solve(env: PickBananaMarginEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    env = env.unwrapped
    FINGER_LENGTH = 0.025

    # Compute grasp pose using OBB of the banana object
    obb = get_actor_obb(env.obj)
    approach_dir = np.array([0, 0, -1])  # Approach from top
    tcp_matrix = env.agent.tcp.pose.to_transformation_matrix()
    tcp_y_axis = tcp_matrix[0, :3, 1].numpy()  # TCP's Y-axis as initial closing direction

    # Orthogonalize closing direction with respect to approach direction
    closing_dir = tcp_y_axis - (approach_dir @ tcp_y_axis) * approach_dir
    closing_dir = closing_dir / np.linalg.norm(closing_dir)

    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approach_dir,
        target_closing=closing_dir,
        depth=FINGER_LENGTH,
    )
    grasp_pose = env.agent.build_grasp_pose(approach_dir, grasp_info["closing"], grasp_info["center"])

    # Move above the banana (approach)
    pre_grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    res = planner.move_to_pose_with_screw(pre_grasp_pose)
    if res == -1: return -1

    # Move down and grasp the banana
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1: return -1
    planner.close_gripper()

    # Lift the banana to the goal position (same orientation as grasp)
    goal_pose = sapien.Pose(p=env.goal_site.pose.sp.p, q=grasp_pose.q)
    res = planner.move_to_pose_with_screw(goal_pose)
    if res == -1: return -1

    planner.close()
    return res