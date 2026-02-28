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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="maze")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    args = parser.parse_args()

    device = "cuda"

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env = MazeEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
        in_eval=True,
        in_debug=args.debug,
    )

    runner = OnPolicyRunnerHRL(env, train_cfg, log_dir, device=device)
    env.runner = runner
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=device)

    obs, _ = env.reset()
    i = 0
    with torch.no_grad():
        while True:
            i += 1
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)
            time.sleep(0.1)

if __name__ == "__main__":
    main()

# python3 environnements/maze/maze_eval.py -e maze_07-02-2026_16:58:49 --ckpt 350