import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.rotate_cube import RotateCubeEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb

def solve(env: RotateCubeEnv, seed=None, debug=False, vis=False):
    # Reset the environment
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    env_u = env.unwrapped
    FINGER_LENGTH = 0.025

    # Calculate the grasping posture
    obb = get_actor_obb(env_u.cube)
    approaching = np.array([0, 0, -1])
    target_closing = env_u.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env_u.agent.build_grasp_pose(approaching, closing, center)

    # Approaching in the -z direction
    pre_grasp = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(pre_grasp)

    # Grasp
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # Rotate around the z-axis to the target angle
    angle_rad = np.deg2rad(env_u.target_angle)
    rot_pose = grasp_pose * sapien.Pose(q=euler2quat(0, 0, angle_rad))
    res = planner.move_to_pose_with_screw(rot_pose)

    # Release
    planner.open_gripper()
    planner.close()
    return res