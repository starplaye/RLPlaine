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
from core.on_policy_runner import OnPolicyRunner
from core.on_policy_runner_hrl import OnPolicyRunnerHRL

from maze_train import get_train_manager_cfg, get_cfgs_manager

MODE = ["RL","HRL"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="maze")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    args = parser.parse_args()

    device = "cuda"

    is_hrl = "HRL" in args.exp_name

    log_dir = f"logs/{args.exp_name}"

    if not is_hrl :
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg= pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
        reward_cfg["reward_scales"] = {}
        
        env = MazeEnv(
            num_envs=2,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=True,
            in_eval=True,
            in_debug=args.debug,
        )

        runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)
        env.runner = runner
        resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=device)

        runner.current_learning_iteration = 8000

        obs = env.reset()
        i = 0
        timer = time.time()
        with torch.no_grad():
            while True:
                i += 1
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)
                time.sleep(1)
                if i%100 == 0:
                    print("=>", 100 / (time.time()-timer))
                    timer = time.time()
                    w,f,r = env.stats_game()
                    print(f"Win {w}% // Regret {r}")
    else :
        train_manager_cfg, manager_cfg, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
        reward_cfg["reward_scales"] = {}
        
        env = MazeEnv(
            num_envs=8192,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False,
            is_manager=True,
            manager_cfg=manager_cfg
        )
        
        runner = OnPolicyRunnerHRL(env, train_manager_cfg, log_dir, device=device)
        env.runner = runner
        resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=device)
        policy_worker = runner.worker_runner.get_inference_policy(device=device)
        
        batch_idx = torch.arange(env.num_envs, device=device)

        obs = env.reset()
        manager_obs, extras = env.get_observations_manager()
        i = 0
        timer = time.time()
        with torch.no_grad():
            while True:
                i += 1
                # print(env.stats_game())

                manager_obs, extras = env.get_observations_manager()
                actions_manager = policy(manager_obs)
                env.set_subgoal(actions_manager,batch_idx)
                
                for _ in range(10):
                    actions = policy_worker(obs)
                    obs, _, _, _, _, _, _, _ = env.step(actions)
                    # time.sleep(1)
                    
                if i%100 == 0:
                    print("=>", 100 / (time.time()-timer))
                    timer = time.time()
                    w,f,r = env.stats_game()
                    print(f"Win {w}% // Regret {r}")

if __name__ == "__main__":
    main()

# python3 environnements/maze/maze_eval.py -e maze_07-02-2026_16:58:49 --ckpt 350