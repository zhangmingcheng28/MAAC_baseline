import os
import torch
import random
from ._base import _Base
from marl.utils import ReplayMemory, Transition, PrioritizedReplayMemory, soft_update, hard_update
from marl.utils import LinearDecay
from torch.nn import MSELoss
import numpy as np
# from ma_gym.wrappers import Monitor


class DQNShareNoConsensus(_Base):
    """
    Independent DQN + Double DQN + Prioritized Replay + Soft Target Updates

    Paper: Multi agent reinforcement learning independent vs cooperative agents
    Url: http://web.media.mit.edu/~cynthiab/Readings/tan-MAS-reinfLearn.pdf
    """

    def __init__(self, env_fn, model_fn, lr, discount, batch_size, device, mem_len, tau, train_episodes,
                 episode_max_steps, path):
        super().__init__(env_fn, model_fn, lr, discount, batch_size, device, train_episodes, episode_max_steps, path)
        self.memory = PrioritizedReplayMemory(mem_len)
        self.tau = tau

        self.target_model = model_fn().to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.share_iter = 4

        self.exploration = LinearDecay(0.1, 1.0, self.train_episodes)
        self.__update_iter = 0

    def _get_critic_consensus(self):
        if self.model.n_agents > 1:

            # add weigths of others to the agent 0
            target = self.model.agent(0)._critic
            for source_i in range(1, self.model.n_agents):
                source = self.model.agent(source_i)._critic
                for target_param, param in zip(target.parameters(), source.parameters()):
                    target_param.data.copy_(target_param.data + param.data)

            # normalize weights of agents 0
            for param in target.parameters():
                param.data.copy_(param.data / self.model.n_agents)

            # copy weight of agent 0 to others
            source = self.model.agent(0)._critic
            for target_i in range(1, self.model.n_agents):
                target = self.model.agent(target_i)._critic
                for target_param, param in zip(target.parameters(), source.parameters()):
                    target_param.data.copy_(param.data)

    def _get_thoughts(self, obs_batch):
        og_thoughts = []
        for agent_i in range(self.model.n_agents):
            og_thoughts.append(self.model.agent(agent_i).get_thought(obs_batch[:, agent_i]))

        global_thoughts = [x.clone() for x in og_thoughts]
        for i in range(self.share_iter):
            for agent_i in range(self.model.n_agents):
                global_thoughts[agent_i] = (global_thoughts[agent_i] +
                                            global_thoughts[(agent_i + 1) % len(global_thoughts)]) / 2
        global_thoughts = torch.stack(global_thoughts)
        og_thoughts = torch.stack(og_thoughts)

        return og_thoughts, global_thoughts

    def __update(self, obs_n, action_n, next_obs_n, reward_n, done):
        self.model.train()

        self.memory.push(obs_n, action_n, next_obs_n, reward_n, done)

        if self.batch_size > len(self.memory):
            self.model.eval()
            return None

        # Todo: move this beta in the Prioritized Replay memory
        beta_start = 0.4
        beta = min(1.0, beta_start + (self.__update_iter + 1) * (1.0 - beta_start) / 5000)

        transitions, indices, weights = self.memory.sample(self.batch_size, beta)
        batch = Transition(*zip(*transitions))

        obs_batch = torch.FloatTensor(list(batch.state)).to(self.device)
        action_batch = torch.FloatTensor(list(batch.action)).to(self.device)
        reward_batch = torch.FloatTensor(list(batch.reward)).to(self.device)
        next_obs_batch = torch.FloatTensor(list(batch.next_state)).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        non_final_mask = 1 - torch.ByteTensor(list(batch.done)).to(self.device)

        # calc loss
        prios = 0
        overall_loss = 0

        og_thoughts, global_thoughts = self._get_thoughts(obs_batch)
        next_obs_og_thoughts, next_obs_global_thoughts = self._get_thoughts(next_obs_batch)
        for i in range(self.model.n_agents):
            q_val_i = self.model.agent(i)(og_thoughts[i], global_thoughts[i])
            pred_q = q_val_i.gather(1, action_batch[:, i].long().unsqueeze(1))

            target_next_obs_q = torch.zeros(pred_q.shape).to(self.device)
            non_final_next_obs_og = next_obs_og_thoughts[i, :][non_final_mask[:, i]]
            non_final_next_obs_global = next_obs_global_thoughts[i, :][non_final_mask[:, i]]

            # Double DQN update
            target_q = 0
            if not (non_final_next_obs_og.shape[0] == 0):
                _max_actions = self.model.agent(i)(non_final_next_obs_og, non_final_next_obs_global)
                _max_actions = _max_actions.max(1, keepdim=True)[1].detach()
                _max_q = self.target_model.agent(i)(non_final_next_obs_og,
                                                    non_final_next_obs_global).gather(1, _max_actions)
                target_next_obs_q[non_final_mask[:, i]] = _max_q

                target_q = target_next_obs_q.detach()

            target_q = (self.discount * target_q) + reward_batch.sum(dim=1, keepdim=True)
            loss = (pred_q - target_q).pow(2) * weights.unsqueeze(1)
            prios += loss + 1e-5
            loss = loss.mean()
            overall_loss += loss
            self.writer.add_scalar('agent_{}/critic_loss'.format(i), loss.item(), self._step_iter)

        # Optimize the model
        self.optimizer.zero_grad()
        overall_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.memory.update_priorities(indices, prios.data.cpu().numpy())
        self.optimizer.step()

        # update target network
        # if (self._step_iter % 100) == 0:
        #     self._get_critic_consensus()
        soft_update(self.target_model, self.model, self.tau)

        # log
        self.writer.add_scalar('_overall/critic_loss', overall_loss, self._step_iter)
        self.writer.add_scalar('_overall/beta', beta, self._step_iter)

        # just keep track of update counts
        self.__update_iter += 1

        # resuming the model in eval mode
        self.model.eval()

        return loss.item()

    def _select_action(self, model, obs_n, explore=False):
        """ selects epsilon greedy action for the state """
        if explore and self.exploration.eps > random.random():
            act_n = self.env.action_space.sample()
        else:
            act_n = []

            og_thoughts, global_thoughts = self._get_thoughts(obs_n)
            for i in range(model.n_agents):
                act_n.append(model.agent(i)(og_thoughts[i], global_thoughts[i]).argmax(1).item())

        return act_n

    def _train(self, episodes):
        self.model.eval()
        train_rewards = []
        train_loss = []

        for ep in range(episodes):
            terminal = False
            obs_n = self.env.reset()
            ep_step = 0
            ep_reward = [0 for _ in range(self.model.n_agents)]
            while not terminal:
                torch_obs_n = torch.FloatTensor(obs_n).to(self.device).unsqueeze(0)
                action_n = self._select_action(self.model, torch_obs_n, explore=True)

                next_obs_n, reward_n, done_n, info = self.env.step(action_n)
                terminal = all(done_n) or ep_step >= self.episode_max_steps

                done_n = [terminal for _ in range(self.env.n_agents)]
                loss = self.__update(obs_n, action_n, next_obs_n, reward_n, done_n)

                obs_n = next_obs_n
                ep_step += 1
                self._step_iter += 1
                if loss is not None:
                    train_loss.append(loss)

                for i, r_n in enumerate(reward_n):
                    ep_reward[i] += r_n

            train_rewards.append(ep_reward)
            self.exploration.update()

            # log - training
            for i, r_n in enumerate(ep_reward):
                self.writer.add_scalar('agent_{}/train_reward'.format(i), r_n, self._step_iter)
            self.writer.add_scalar('_overall/train_reward', sum(ep_reward), self._step_iter)
            self.writer.add_scalar('_overall/train_ep_steps', ep_step, self._step_iter)
            self.writer.add_scalar('_overall/exploration_rate', self.exploration.eps, self._step_iter)

            print(ep, sum(ep_reward))

        return np.array(train_rewards).mean(axis=0), (np.mean(train_loss) if len(train_loss) > 0 else [])

    def test(self, episodes, render=False, log=False, record=False):
        self.model.eval()
        env = self.env
        # if record:
        #     env = Monitor(self.env_fn(), directory=os.path.join(self.path, 'recordings'), force=True,
        #                   video_callable=lambda episode_id: True)
        with torch.no_grad():
            test_rewards = []
            total_test_steps = 0
            for ep in range(episodes):
                terminal = False
                obs_n = env.reset()
                step = 0
                ep_reward = [0 for _ in range(self.model.n_agents)]
                while not terminal:
                    if render:
                        env.render()

                    torch_obs_n = torch.FloatTensor(obs_n).to(self.device).unsqueeze(0)
                    action_n = self._select_action(self.model, torch_obs_n, explore=False)

                    next_obs_n, reward_n, done_n, info = env.step(action_n)
                    terminal = all(done_n) or step >= self.episode_max_steps

                    obs_n = next_obs_n
                    step += 1
                    for i, r_n in enumerate(reward_n):
                        ep_reward[i] += r_n

                total_test_steps += step
                test_rewards.append(ep_reward)

            test_rewards = np.array(test_rewards).mean(axis=0)
            if log:
                # log - test
                for i, r_n in enumerate(test_rewards):
                    self.writer.add_scalar('agent_{}/eval_reward'.format(i), r_n, self._step_iter)
                self.writer.add_scalar('_overall/eval_reward', sum(test_rewards), self._step_iter)
                self.writer.add_scalar('_overall/test_ep_steps', total_test_steps / episodes, self._step_iter)
        if record:
            env.close()

        return test_rewards
