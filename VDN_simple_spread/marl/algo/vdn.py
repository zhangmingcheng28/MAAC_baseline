import torch
import random
from ._base import _Base
from marl.utils import ReplayMemory, Transition, PrioritizedReplayMemory, soft_update, hard_update
from marl.utils import LinearDecay
from torch.nn import MSELoss
import datetime
import time
import wandb
import numpy as np

# Set the default device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)

class VDN(_Base):
    """
    Value Decomposition Network + Double DQN + Prioritized Replay + Soft Target Updates

    Paper: https://arxiv.org/pdf/1706.05296.pdf
    """

    def __init__(self, env_fn, model_fn, lr, discount, batch_size, device, mem_len, tau, train_episodes,
                 episode_max_steps, path):
        super().__init__(env_fn, model_fn, lr, discount, batch_size, device, train_episodes, episode_max_steps, path)
        self.memory = PrioritizedReplayMemory(mem_len)
        self.tau = tau

        self.target_model = model_fn().to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.exploration = LinearDecay(0.1, 1.0, self.train_episodes)
        self._update_iter = 0

    def _reward_from_state(self, n_state, all_agents):
        n_state = [n_state[0, i] for i in range(n_state.shape[1])]
        rew = []

        for state in n_state:

            obs_landmark = np.array(state[4:10])
            agent_reward = 0
            potential_other = []
            for i in range(3):

                sub_obs = obs_landmark[i * 2: i * 2 + 2]
                dist = np.sqrt(sub_obs[0] ** 2 + sub_obs[1] ** 2)

                # if dist < 0.4: agent_reward += 0.3
                if dist < 0.2: agent_reward += 0.5
                if dist < 0.1: agent_reward += 1.

            otherA = np.array(state[10:12])  # original
            otherB = np.array(state[12:14])  # original

            dist = np.sqrt(otherA[0] ** 2 + otherA[1] ** 2)
            if dist < 3.1:  agent_reward -= 0.25
            dist = np.sqrt(otherB[0] ** 2 + otherB[1] ** 2)
            if dist < 3.1:  agent_reward -= 0.25

            rew.append(agent_reward)

        return rew

    def __update(self, obs_n, action_n, next_obs_n, reward_n, done):
        self.model.train()

        self.memory.push(obs_n, action_n, next_obs_n, reward_n, done)

        if self.batch_size > len(self.memory):
            self.model.eval()
            return None

        # Todo: move this beta in the Prioritized Replay memory
        beta_start = 0.4
        beta = min(1.0, beta_start + (self._update_iter + 1) * (1.0 - beta_start) / 5000)

        transitions, indices, weights = self.memory.sample(self.batch_size, beta)
        batch = Transition(*zip(*transitions))

        obs_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.FloatTensor(np.array(batch.action)).to(self.device)
        reward_batch = torch.FloatTensor(np.array(batch.reward)).to(self.device)
        next_obs_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        # non_final_mask = 1 - torch.ByteTensor(list(batch.done)).to(self.device)
        non_final_mask = ~torch.tensor(batch.done, dtype=torch.bool).to(self.device)

        # calc loss
        overall_pred_q, target_q = 0, 0
        for i in range(self.model.n_agents):
            q_val_i = self.model.agent(i)(obs_batch[:, i])
            overall_pred_q += q_val_i.gather(1, action_batch[:, i].long().unsqueeze(1))

            target_next_obs_q = torch.zeros(overall_pred_q.shape).to(self.device)
            non_final_next_obs_batch = next_obs_batch[:, i][non_final_mask]

            # Double DQN update
            if not (non_final_next_obs_batch.shape[0] == 0):
                _max_actions = self.model.agent(i)(non_final_next_obs_batch).max(1, keepdim=True)[1].detach()
                _max_q = self.target_model.agent(i)(non_final_next_obs_batch).gather(1, _max_actions)
                target_next_obs_q[non_final_mask] = _max_q

                target_q += target_next_obs_q.detach()

        target_q = (self.discount * target_q) + reward_batch.sum(dim=1, keepdim=True)
        loss = (overall_pred_q - target_q).pow(2) * weights.unsqueeze(1)
        prios = loss + 1e-5
        loss = loss.mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.memory.update_priorities(indices, prios.data.cpu().numpy())
        self.optimizer.step()

        # update target network
        # Todo: Make 100 as a parameter
        # if self.__update_iter % 100:
        #     hard_update(self.target_model, self.model)
        if self._update_iter % 5:  # do a soft update every 5 steps, after we have enough in experience replay.
            soft_update(self.target_model, self.model, self.tau)

        # log
        # self.writer.add_scalar('_overall/critic_loss', loss, self._step_iter)
        # self.writer.add_scalar('_overall/beta', beta, self._step_iter)

        # just keep track of update counts
        self._update_iter += 1

        # resuming the model in eval mode
        self.model.eval()

        return loss.item()

    def _select_action(self, model, obs_n, explore=False):
        """ selects epsilon greedy action for the state """
        if explore and self.exploration.eps > random.random():
            # act_n = self.env.action_space.sample()
            act_n = [space.sample() for space in self.env.action_space]
            # act_n = [np.eye(space.n)[np.random.choice(space.n)] for space in self.env.action_space]
        else:
            act_n = []
            for i in range(model.n_agents):
                act_n.append(model.agent(i)(obs_n[:, i]).argmax(1).item())

        return act_n

    def _train(self, episodes):

        today = datetime.date.today()
        current_date = today.strftime("%d%m%y")
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%H_%M_%S")

        wandb.login(key="efb76db851374f93228250eda60639c70a93d1ec")
        wandb.init(
            # set the wandb project where this run will be logged
            project="MADDPG_sample_newFrameWork",
            name='VDN_' + device.type + '_SS_D_test_' + str(current_date) + '_' + str(formatted_time),
            # track hyperparameters and run metadata
            config={
                "epochs": episodes,
            }
        )

        self.model.eval()
        train_rewards = []
        train_loss = []
        ep_acc_rws = 0

        for ep in range(episodes):
            terminal = False
            obs_n = self.env.reset()
            obs_n = [obs_n[0, i, :] for i in range(obs_n.shape[1])]
            ep_step = 0
            ep_reward = [0 for _ in range(self.model.n_agents)]
            ep_acc_rws = 0
            eps_start_time = time.time()
            while not terminal:
                torch_obs_n = torch.FloatTensor(np.array(obs_n)).to(self.device).unsqueeze(0)
                action_n = self._select_action(self.model, torch_obs_n, explore=True)  # eps-greedy here.

                # -------- amend action_n to one-hot form ---------
                # so that can be feed into simple_spread environment
                action_n_adjusted = [[np.eye(self.env.action_space[0].n)[idx]for idx in action_n]]  # must have an outter list to ensure algorithm run correctly in this version
                # --------end of amend action_n to one-hot form ---------

                next_obs_n_convert, reward_n_convert, done_n_convert, info = self.env.step(action_n_adjusted)
                # adds additional global reward
                rew1 = self._reward_from_state(next_obs_n_convert, self.env.envs[0].agents)
                true_reward_n_convert = rew1 + (np.array(reward_n_convert, dtype=np.float32) / 100.)
                # config observation to type that can be input into experience replay and training.
                next_obs_n = [next_obs_n_convert[0, i, :] for i in range(next_obs_n_convert.shape[1])]

                reward_n = true_reward_n_convert.flatten().tolist()
                done_n = done_n_convert.flatten().tolist()
                terminal = all(done_n) or ep_step >= self.episode_max_steps
                # if ep_step >= self.episode_max_steps:
                #     terminal = True

                loss = self.__update(obs_n, action_n, next_obs_n, reward_n, terminal)

                obs_n = next_obs_n
                ep_step += 1
                self._step_iter += 1
                if loss is not None:
                    train_loss.append(loss)

                # for i, r_n in enumerate(reward_n):
                #     ep_reward[i] += r_n
                ep_acc_rws = ep_acc_rws + sum(reward_n)

            train_rewards.append(ep_reward)
            self.exploration.update()  # update eps greedy

            # # log - training
            # for i, r_n in enumerate(ep_reward):
            #     self.writer.add_scalar('agent_{}/train_reward'.format(i), r_n, self._step_iter)
            # self.writer.add_scalar('_overall/train_reward', sum(ep_reward), self._step_iter)
            # self.writer.add_scalar('_overall/train_ep_steps', ep_step, self._step_iter)
            # self.writer.add_scalar('_overall/exploration_rate', self.exploration.eps, self._step_iter)

            # print(ep, sum(ep_reward))
            # print(ep, ep_acc_rws)
            eps_end = time.time() - eps_start_time
            # print("accumulated episode is {} reward is {}, time used is {} seconds".format(ep, ep_acc_rws, eps_end))
            print("Current episode is {}, episode reward is {}, current eps is {}".format(ep, ep_acc_rws, self.exploration.eps))
            # print("accumulated episode is {} reward is {},".format(ep, ep_acc_rws))
            wandb.log({'episode_rewards': float(ep_acc_rws)})
        wandb.finish()
        # return np.array(train_rewards).mean(axis=0), (np.mean(train_loss) if len(train_loss) > 0 else [])
        return ep_acc_rws, (np.mean(train_loss) if len(train_loss) > 0 else [])
