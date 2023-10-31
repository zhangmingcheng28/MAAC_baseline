import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.attention_sac import AttentionSAC
import wandb
import pickle
import time
import datetime

# Set the default device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)

def make_parallel_env(env_id, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=True)
            env._seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def reward_from_state(n_state, all_agents):
    n_state = [n_state[0, i] for i in range(n_state.shape[1])]
    rew = []

    for state in n_state:

        obs_landmark = np.array(state[4:10])
        agent_reward = 0
        potential_other = []
        for i in range(3):

            sub_obs = obs_landmark[i*2: i*2+2]
            dist = np.sqrt(sub_obs[0]**2 + sub_obs[1]**2)

            # if dist < 0.4: agent_reward += 0.3
            if dist < 0.2: agent_reward += 0.5
            if dist < 0.1: agent_reward += 1.


        otherA = np.array(state[10:12])  # original
        otherB = np.array(state[12:14])  # original


        # ----------self added ------------------ #
        # cur_pos = state[2:4]
        # potential_other.append(cur_pos - all_agents[0].state.p_pos)
        # potential_other.append(cur_pos - all_agents[1].state.p_pos)
        # potential_other.append(cur_pos - all_agents[2].state.p_pos)
        # idx_holder = []
        # for items_idx, items in enumerate(potential_other):
        #     if sum(items) == 0:
        #         continue
        #     idx_holder.append(items_idx)
        # otherA = potential_other[idx_holder[0]]
        # otherB = potential_other[idx_holder[1]]
        # ---------- end of self added ---------- #
        dist = np.sqrt(otherA[0] ** 2 + otherA[1] ** 2)
        if dist < 3.1:  agent_reward -= 0.25
        dist = np.sqrt(otherB[0] ** 2 + otherB[1] ** 2)
        if dist < 3.1:  agent_reward -= 0.25

        rew.append(agent_reward)

    return rew


def run(config):
    today = datetime.date.today()
    current_date = today.strftime("%d%m%y")
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%H_%M_%S")

    wandb.login(key="efb76db851374f93228250eda60639c70a93d1ec")
    wandb.init(
        # set the wandb project where this run will be logged
        project="MADDPG_sample_newFrameWork",
        name='MAAC_' + device.type + '_SS_D_test_' + str(current_date) + '_' + str(formatted_time),
        # track hyperparameters and run metadata
        config={
            "epochs": config.n_episodes,
        }
    )

    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(run_num)
    np.random.seed(run_num)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, run_num)
    model = AttentionSAC.init_from_env(env,
                                       tau=config.tau,
                                       pi_lr=config.pi_lr,
                                       q_lr=config.q_lr,
                                       gamma=config.gamma,
                                       pol_hidden_dim=config.pol_hidden_dim,
                                       critic_hidden_dim=config.critic_hidden_dim,
                                       attend_heads=config.attend_heads,
                                       reward_scale=config.reward_scale)
    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0
    eps_reward = []
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset()
        model.prep_rollouts(device=device)
        ep_acc_rws = 0
        eps_start_time = time.time()
        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(model.nagents)]
            # get actions as torch Variables
            torch_agent_actions = model.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() if device=='cpu' else ac.data.cpu().numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)

            # adds additional global reward
            rew1 = reward_from_state(next_obs, env.envs[0].agents)
            rewards = rew1 + (np.array(rewards, dtype=np.float32) / 100.)

            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)

            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and (t % config.steps_per_update) < config.n_rollout_threads):
                if config.use_gpu:
                    model.prep_training(device=device)
                else:
                    model.prep_training(device=device)
                for u_i in range(config.num_updates):
                    sample = replay_buffer.sample(config.batch_size,
                                                  to_gpu=config.use_gpu)
                    model.update_critic(sample, logger=logger)
                    model.update_policies(sample, logger=logger)
                    model.update_all_targets()
                model.prep_rollouts(device=device)
            ep_acc_rws = ep_acc_rws + sum(rewards[0])  # must sum(rewards[0]), because rewards is (1x3) not (3,) in shape
        # ep_rews = replay_buffer.get_average_rewards(config.episode_length * config.n_rollout_threads)

        # for a_i, a_ep_rew in enumerate(ep_acc_rws):
        #     logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew * config.episode_length, ep_i)
        eps_end = time.time() - eps_start_time
        print("accumulated episode reward is {}, time used is {} seconds".format(ep_acc_rws, eps_end))
        eps_reward.append(ep_acc_rws)
        # save the reward for pickle.
        with open(str(run_dir) + '/all_episode_reward.pickle', 'wb') as handle:
            pickle.dump(eps_reward, handle, protocol=pickle.HIGHEST_PROTOCOL)
        wandb.log({'episode_rewards': float(ep_acc_rws)})

        # if ep_i % config.save_interval < config.n_rollout_threads:
        #     model.prep_rollouts(device='cpu')
        #     os.makedirs(run_dir / 'incremental', exist_ok=True)
        #     model.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
        #     model.save(run_dir / 'model.pt')
        if ep_i % config.save_interval == 0:
            model.save(run_dir / 'model.pt')

    model.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", default="simple_spread.py", help="Name of environment")
    parser.add_argument("model_name", help="Name of directory to store " + "model/training contents")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=50000, type=int)
    parser.add_argument("--episode_length", default=50, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)  # update interval, originally is 100, the smaller this number the longer time to complete one episode
    parser.add_argument("--num_updates", default=4, type=int, help="Number of updates per update cycle")  # was 4
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size for training")  # batch size was 1024
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)  # actor lr, was 0.001
    parser.add_argument("--q_lr", default=0.001, type=float)  # critic lr, was 0.001
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.95, type=float)  # was 0.99
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--use_gpu", action='store_true')

    config = parser.parse_args()

    run(config)
