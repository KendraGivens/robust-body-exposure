import gym, sys, argparse
import numpy as np
from .learn import make_env
import os.path
import pathlib
import time
import pybullet as p
# import assistive_gym

if sys.version_info < (3, 0):
    print('Please use Python 3')
    exit()

def sample_action(env, coop):
    if coop:
        return {'robot': env.action_space_robot.sample(), 'human': env.action_space_human.sample()}
    return env.action_space.sample()

def viewer(env_name):
    coop = 'Human' in env_name
    seed = 18237961019803109524
    target = 4
    env = make_env(env_name, coop=True, seed=seed) if coop else gym.make(env_name)

    env.set_env_variations(
        collect_data = False,
        blanket_pose_var = False,
        high_pose_var = False,
        body_shape_var = False)

    env.set_singulate(True)
    env.set_recover(False)
    env.set_target_limb_code(target)
    env.set_seed_val(seed)

    num_rollouts = 2

    for i in range(num_rollouts):
        done = False
        env.render()
        observation = env.reset()
        env.set_iteration(seed)

        uncover_action = sample_action(env, coop)
        recover_action = sample_action(env, coop)

        #Perform same action every time
        uncover_action = np.array([ 0.77959553,  0.680941  ,  0.19683735, -0.17073987])
        recover_action = np.array([-0.21,  0.49,  0.66,  0.47])

        # uncover_action = np.array([ 0,0,0,0])
        # recover_action = np.array([-0.21,  0.49,  0.66,  0.47])

        while not done:
            cloth_intial, cloth_intermediate, _ = env.uncover_step(uncover_action)
            cloth_final, _ = env.recover_step(recover_action)
            observation, uncover_reward, recover_reward, done, info = env.get_info()

            if coop:
                done = done['__all__']

        # print(f"Trial {i}: Uncover Reward = {uncover_reward:.2f} Recover Reward = {recover_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
    parser.add_argument('--env', default='RobeReversible-v1')
    args = parser.parse_args()

    viewer(args.env)
