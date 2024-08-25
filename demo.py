import argparse
import os
import time
from str2bool import str2bool # type: ignore
import random
import numpy as np
import torch
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="CartPole-v1",
                        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument('--seed', type=int, default=42,
                        help="seed of the experiment")
    parser.add_argument('--total-timesteps', type=int, default=2000000,
                        help="total timesteps of the experiments")
    parser.add_argument('--torch-deterministic', type=lambda x:str2bool(x), default=True, nargs='?', const=True,
                        help="if toggled, torch.backends.cudnn.deterministic=False")
    parser.add_argument('--cuda', type=lambda x:str2bool(x), default=True, nargs='?', const=True,
                        help="if toggled, cuda will not be enabled by default")
    parser.add_argument('--track', type=lambda x:str2bool(x), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument('--capture-video', type=lambda x:str2bool(x), default=False, nargs="?", const=True,
                        help="whether to capture videos of the agent performances (check out `videos` folder)")
    
    # Algorithm specific arguments
    parser.add_argument('--num-envs', type=int, default=4,
                        help="the number of parallel game environments")
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    # print(args)
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|params|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
    )

    ## SEEDING
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    # env = gym.make(args.gym_id, render_mode="rgb_array")
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = gym.wrappers.RecordVideo(env, video_folder="videos", episode_trigger=lambda x: x % 100 == 0)
    # observation = env.reset()
    # episodic_return = 0
    # for _ in range(200):
    #     action = env.action_space.sample()
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     episodic_return += reward
    #     if terminated or truncated:
    #         observation, info = env.reset()
    #         # writer.add_scalar("episodic_return", episodic_return, global_step=episodic_return)
    #         # print(f"episodic_return {info}")
    #         print(f"episodic_return {episodic_return}")
    #         episodic_return = 0
    # env.close()

    writer.close()

    def make_env(gym_id, seed, idx, capture_video, run_name):
        def thunk():
            env = gym.make(gym_id, render_mode="rgb_array")
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if capture_video:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, video_folder="videos", episode_trigger=lambda x: x % 100 == 0)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk
    
    envs = gym.vector.SyncVectorEnv([make_env(args.gym_id, seed=42, idx=0, capture_video=False, run_name=run_name)])
    observation = envs.reset()
    for _ in range(500):
        action = envs.action_space.sample()
        observation, reward, terminated, truncated, info = envs.step(action)
        for item in info:
            if isinstance(info[item][0], dict):
                print(*info[item][0]["episode"]['r'])
            # if "episode" in info[item][0]:
                # print(info[item][0]["episode"]['r'])
    
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)])

    envs.close()