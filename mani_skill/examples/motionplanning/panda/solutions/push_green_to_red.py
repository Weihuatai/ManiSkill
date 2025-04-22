import numpy as np
import sapien

from mani_skill.envs.tasks import PushGreenToRedEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver

def solve(env: PushGreenToRedEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped
    planner.close_gripper()
    reach_pose = sapien.Pose(p=env.green_cube.pose.sp.p + np.array([-0.05, 0, 0]), q=env.agent.tcp.pose.sp.q)
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Move and contact the red cube
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(p=env.red_cube.pose.sp.p + np.array([-0.07, 0, 0]),q=env.agent.tcp.pose.sp.q)
    res = planner.move_to_pose_with_screw(goal_pose)

    planner.close()
    return res
