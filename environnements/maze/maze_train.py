import argparse
import os
import sys
import pickle
import shutil
from datetime import datetime
from importlib import metadata
import time
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from maze_env import MazeEnv
from core.on_policy_runner_hrl import OnPolicyRunnerHRL

def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [256, 128, 128],
            "critic_hidden_dims": [256, 128, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 10,
        "save_interval": 50,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 4,
        "maze_data": "environnements/maze/mazer_model/train_data.csv",
        "maze_folder": "environnements/maze/mazer_model/train/",
        # memory
        "memory_size": 10,
        # vision
        "vision_range": 1, # => 3x3 
    }
    obs_cfg = {
        "num_obs": 35,
        "obs_scales": {
            "rel_pos": 1 / 2.0,
        },
    }
    reward_cfg = {
        "timeout": 400, # step
        "base_difficulty" : 0,
        "max_difficulty" : 10,
        "reward_scales": {
            "distance": 1,
            "target": 50,
            "timeout": 50,
            "movement": 10,
            # "already": 10,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "pos_x_range": [-2, 0],
        "pos_y_range": [-2,2],
        "pos_z_range": [0.6, 0.8],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="maze")
    parser.add_argument("-B", "--num_envs", type=int, default=8192) # 4096
    parser.add_argument("--max_iterations", type=int, default=101)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    log_dir = f"logs/{args.exp_name}_{now}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = MazeEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=args.vis
    )
    
    runner = OnPolicyRunnerHRL(env, train_cfg, log_dir, device="cuda")
    env.runner = runner

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    
    # -- Env tester --
    # pos = torch.zeros((args.num_envs, 2), dtype=torch.long, device="cuda")
    # pos[:, 1] = 1  
    # for i in range(1000000):
    #     if i % 100 == 0 : print(i)
    #     env.step(torch.randint(-1,2,(args.num_envs, 2), device="cuda"))


if __name__ == "__main__":
    main()

"""
# training
python3 environnements/maze/maze_train.py --max_iterations 500
"""
