import os
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter
import numpy as np
# from ma_gym.wrappers import Monitor


class _Base:
    """ Base Class for  Multi Agent Algorithms"""

    def __init__(self, env_fn, model_fn, lr, discount, batch_size, device, train_episodes, episode_max_steps, path):
        """

        Args:
            env_fn:
            model_fn:
            lr:
            discount:
            batch_size:
            device:
            train_episodes:
            episode_max_steps:
            path:
            log_suffix:
        """
        self.env_fn = env_fn
        self.env = env_fn()
        # self.env._seed(777)  # Todo: Add seed to attributes
        self.train_episodes = train_episodes
        self.episode_max_steps = episode_max_steps

        self.model = model_fn().to(device)
        self.lr = lr
        self.discount = discount
        self.batch_size = batch_size
        self.device = device

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # logging + visualization
        self.path = path
        self.best_model_path = os.path.join(self.path, 'model.p')
        self.last_model_path = os.path.join(self.path, 'last_model.p')
        self.writer = None
        self._step_iter = 0  # total environment steps

    def save(self, path):
        """ save relevant properties in given path"""
        torch.save(self.model.state_dict(), path)

    def restore(self, path=None):
        """
        Restores the model from the given path

        Args:
            path (optional) : model path

        """
        path = self.best_model_path if path is None else path
        self.model.load_state_dict(torch.load(path))

    def __writer_close(self):
        self.writer.export_scalars_to_json(os.path.join(self.path, 'summary.json'))
        self.writer.close()
        print('saved')

    def close(self):
        """ It should be called after one is done with the usage"""
        self.env.close()

    def _select_action(self, model, obs_n, explore=False):
        """ selects epsilon greedy action for the state """
        raise NotImplementedError

    def _train(self, test_interval):
        raise NotImplementedError

    def _reward_from_state(self, n_state, all_agents):  # just to ensure the functions below is not bolded in yellow
        raise NotImplementedError

    def train(self, test_interval=50):

        print('Training......')

        train_score, train_loss = self._train(self.train_episodes)  # we train for "self.train_episodes".

        # keeping a copy of last trained model
        print('all training done, saving model')
        self.save(self.last_model_path)

    def test(self, episodes, render=False, log=False, record=False):
        self.model.eval()
        env = self.env
        # if record:
        #     env = Monitor(self.env_fn(), directory=os.path.join(self.path, 'recordings'), force=True,
        #                   video_callable=lambda episode_id: True)
        with torch.no_grad():
            test_rewards = []
            total_test_steps=0
            for ep in range(episodes):
                terminal = False
                obs_n = env.reset()
                obs_n = [obs_n[0, i, :] for i in range(obs_n.shape[1])]
                step = 0
                ep_reward = [0 for _ in range(self.model.n_agents)]
                ep_acc_rws = 0
                while not terminal:
                    if render:
                        env.render()

                    torch_obs_n = torch.FloatTensor(np.array(obs_n)).to(self.device).unsqueeze(0)
                    action_n = self._select_action(self.model, torch_obs_n, explore=False)

                    # -------- amend action_n to one-hot form ---------
                    # so that can be feed into simple_spread environment
                    action_n_adjusted = [[np.eye(self.env.action_space[0].n)[idx]for idx in action_n]]
                    # --------end of amend action_n to one-hot form ---------
                    next_obs_n_convert, reward_n_convert, done_n_convert, info = env.step(action_n_adjusted)

                    done_n = done_n_convert.flatten().tolist()
                    next_obs_n = [next_obs_n_convert[0, i, :] for i in range(next_obs_n_convert.shape[1])]

                    rew1 = self._reward_from_state(next_obs_n_convert, self.env.envs[0].agents)
                    true_reward_n_convert = rew1 + (np.array(reward_n_convert, dtype=np.float32) / 100.)
                    reward_n = true_reward_n_convert.flatten().tolist()

                    terminal = all(done_n) or step >= self.episode_max_steps

                    obs_n = next_obs_n
                    step += 1
                    # for i, r_n in enumerate(reward_n):
                    #     ep_reward[i] += r_n
                    ep_acc_rws = ep_acc_rws + sum(reward_n)

                # total_test_steps += step
                # test_rewards.append(ep_reward)

            # test_rewards = np.array(test_rewards).mean(axis=0)
        #     if log:
        #         # log - test
        #         for i, r_n in enumerate(test_rewards):
        #             self.writer.add_scalar('agent_{}/eval_reward'.format(i), r_n, self._step_iter)
        #         self.writer.add_scalar('_overall/eval_reward', sum(test_rewards), self._step_iter)
        #         self.writer.add_scalar('_overall/test_ep_steps', total_test_steps / episodes, self._step_iter)
        # if record:
        #     env.close()

        return ep_acc_rws
