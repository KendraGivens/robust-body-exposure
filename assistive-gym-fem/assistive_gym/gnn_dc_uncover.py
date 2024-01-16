import sys, argparse, multiprocessing, time, os, math
import numpy as np
import pickle, pathlib
import os.path as osp
import random
import glob
from pathlib import Path import gym
from gym.utils import seeding
from learn import make_env
from assistive_gym.envs.bu_gnn_util import scale_action, check_grasp_on_cloth

uncover_model_path = '/home/kpputhuveetil/git/robe/robust-body-exposure/trained_models/FINAL_MODELS/standard_2D_not_subsampled_epochs=250_batch=100_workers=4_1687986938'
eval_dir_name = 'cma_evaluations'
threshold = 0.745

search_dir = osp.join(uncover_model_path, eval_dir_name)

eval_conditions = ['TL_[2, 4, 5, 8, 10, 11, 12, 13, 14, 15]_Uncover_Evals_Train_New_Grasp_Parsed']

data_path = osp.join(uncover_model_path, eval_dir_name, eval_conditions[0], 'raw/')
filenames = list(Path(data_path).glob('*.pkl'))

def sample_action(env):
    return env.action_space.sample()

def gnn_data_collect(env_name, i, target):
    coop = 'Human' in env_name
    on_grasp = False
    uncover_action = []
    recover_action = []
    filename = []
    recover = False

    seed = seeding.create_seed()

    # create environment
    env = make_env(env_name, coop=coop, seed=seed)
    env.set_env_variations(
        collect_data = True,
        blanket_pose_var = False,
        high_pose_var = False,
        body_shape_var = False)

    env.set_singulate(True)
    env.set_target_limb_code(target)
    env.set_recover(recover)
    env.set_seed_val(seed)

    done = False
    observation = env.reset()
    pid = os.getpid()

    # set up actions
    uncover_action = sample_action(env)
    recover_action = [0, 0, 0, 0]

    cloth_initial_sim, cloth_intermediate_sim, execute_uncover_action = env.uncover_step(uncover_action)

    if not execute_uncover_action:
        return [filename, pid]

    cloth_final_sim, execute_recover_action = env.recover_step(recover_action)
    observation, uncover_reward, recover_reward, done, info = env.get_info()

    if not recover:
        recover_action = []

    filename = f"c_{target}_{seed}_pid{pid}"
    with open(osp.join(pkl_loc, filename +".pkl"),"wb") as f:
        pickle.dump({
            "recovering":recover,
            "observation":observation,
            "info":info,
            "uncover_action":uncover_action,
            "recover_action":recover_action}, f)

    output = [i, target, pid]
    del env

    return output

def counter_callback(output):
    global counter
    counter += 1
    print(f"Trial Completed: {output[0]}, Worker: {os.getpid()}, Target: {output[1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data collection for gnn training')
    parser.add_argument('--env', default='RobeReversible-v1')
    parser.add_argument('--num_seeds', type=int, default=100)
    parser.add_argument('--rollouts', type=int, default=10000)
    parser.add_argument('--target_limb_list', type=str, default='2, 4, 5, 8, 10, 11, 12, 13, 14, 15')
    args = parser.parse_args()

    target_limb_list = [int(item) for item in args.target_limb_list.split(',')]
    target_list = target_limb_list * 1000

    current_dir = os.getcwd()
    recover = False

    recover_string = 'Uncover_Data'
    if recover:
        recover_string = 'Recover_Data'

    variation_type = f'TL_{args.target_limb_list}_{recover_string}_{args.num_seeds}_seeds_{args.rollouts}_states' # for uncovering states are the random actions, for recovering states are = num_seeds

    pkl_loc = os.path.join(current_dir, "DATASETS",recover_string, variation_type, 'raw')
    pathlib.Path(pkl_loc).mkdir(parents=True, exist_ok=True)

    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    counter = 0
    num_processes = 100
    trials = args.rollouts
    counter = 0
    result_objs = []

    for j in range(math.ceil(trials/num_processes)):
        with multiprocessing.Pool(processes=num_processes) as pool:
            for i in range(num_processes):
                target = target_list[i+i*j]
                result = pool.apply_async(gnn_data_collect, args = (args.env, i, target), callback=counter_callback)
                result_objs.append(result)
            results = [result.get() for result in result_objs]