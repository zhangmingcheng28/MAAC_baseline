import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from utils.agents import AttentionAgent
from utils.critics import AttentionCritic
import copy
import numpy as np
import time

MSELoss = torch.nn.MSELoss()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class AttentionSAC(object):
    """
    Wrapper class for SAC agents with central attention critic in multi-agent
    task
    """
    def __init__(self, agent_init_params, sa_size,
                 gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01,
                 reward_scale=10.,
                 pol_hidden_dim=128,
                 critic_hidden_dim=128, attend_heads=4,
                 **kwargs):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
            sa_size (list of (int, int)): Size of state and action space for
                                          each agent
            gamma (float): Discount factor
            tau (float): Target update rate
            pi_lr (float): Learning rate for policy
            q_lr (float): Learning rate for critic
            reward_scale (float): Scaling for reward (has effect of optimal
                                  policy entropy)
            hidden_dim (int): Number of hidden dimensions for networks
        """
        self.nagents = len(sa_size)

        self.agents = [AttentionAgent(lr=pi_lr,
                                      hidden_dim=pol_hidden_dim,
                                      **params)
                         for params in agent_init_params]
        self.critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim,
                                      attend_heads=attend_heads)
        self.target_critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim,
                                             attend_heads=attend_heads)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr,
                                     weight_decay=1e-3)
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.reward_scale = reward_scale
        self.pol_dev = device  # device for policies
        self.critic_dev = device  # device for critics
        self.trgt_pol_dev = device  # device for target policies
        self.trgt_critic_dev = device  # device for target critics
        self.niter = 0

        self.var = [1.0 for i in range(len(agent_init_params))]  # added for con act space

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def step(self, observations, ep_i, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
        Outputs:
            actions: List of actions for each agent
        """
        # ---------- added for con act space-------------
        outlist = []
        for a, obs, var_idx in zip(self.agents, observations, range(len(self.var))):
            act = a.step(obs, explore=explore)
        # ------- for no extra noise -------------
        #     act = torch.clamp(act, -1.0, 1.0)
            outlist.append(act)
        # ------- end for no extra noise -------------
        #     if explore:
        #         act = act + torch.from_numpy(np.random.randn(2) * self.var[var_idx])
        #         if self.var[var_idx] > 0.05:
        #             self.var[var_idx] = self.var[var_idx] * 0.999998
        #         act = torch.clamp(act, -1.0, 1.0)
        #     outlist.append(act)
        # for var_idx, var in enumerate(self.var):
        #     print("Current episode {}, agent {} var is {}".format(ep_i, var_idx, var))
        return outlist
        # end of adding for con act space --------------------

        # outlist=[]
        # for a, obs in zip(self.agents, observations):
        #     outlist.append(a.step(obs, explore=explore))
        # return outlist


    def update_critic(self, sample, soft=True, logger=None, **kwargs):
        """
        Update central critic for all agents
        """
        obs, acs, rews, next_obs, dones = sample
        # Q loss
        next_acs = []
        next_log_pis = []
        for pi, ob in zip(self.target_policies, next_obs):
            # curr_next_ac, curr_next_log_pi = pi(ob, return_log_pi=True)  # pi is the target actor policy
            curr_next_ac, curr_next_log_pi = pi(ob, return_log_pi=True)  # try if we just output continuous action
            next_acs.append(curr_next_ac)
            next_log_pis.append(curr_next_log_pi)
        trgt_critic_in = list(zip(next_obs, next_acs))
        critic_in = list(zip(obs, acs))
        next_qs = self.target_critic(trgt_critic_in)
        critic_rets = self.critic(critic_in, regularize=True, logger=logger, niter=self.niter)
        q_loss = 0
        # for a_i, nq, log_pi, (pq, regs) in zip(range(self.nagents), next_qs, next_log_pis, critic_rets):
        #     target_q = (rews[a_i].view(-1, 1) + self.gamma * nq * (1 - dones[a_i].view(-1, 1)))
        #     if soft:  # this is a technique that is used in SAC. Used to calculate entropy regularization term
        #         target_q -= log_pi / self.reward_scale
        #     q_loss += MSELoss(pq, target_q.detach())
        #     for reg in regs:
        #         q_loss += reg  # regularizing attention

        # no regularizing for attention critic output loss or the Q loss.
        for a_i, nq, log_pi, (pq, regs) in zip(range(self.nagents), next_qs, next_log_pis, critic_rets):
            target_q = (rews[a_i].view(-1, 1) + self.gamma * nq * (1 - dones[a_i].view(-1, 1)))
            if soft:
                target_q -= log_pi / self.reward_scale
            # q_loss = q_loss + MSELoss(pq, target_q.detach())
            q_loss += MSELoss(pq, target_q.detach())
            for reg in regs:
                q_loss += reg  # regularizing attention
        q_loss.backward()
        # self.critic.scale_shared_grads()
        # grad_norm = torch.nn.utils.clip_grad_norm(self.critic.parameters(), 10 * self.nagents)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        # if logger is not None:
        #     logger.add_scalar('losses/q_loss', q_loss, self.niter)
        #     logger.add_scalar('grad_norms/q', grad_norm, self.niter)
        self.niter += 1

    def update_policies(self, sample, soft=True, logger=None, **kwargs):
        # soft = False
        obs, acs, rews, next_obs, dones = sample
        samp_acs = []
        # all_probs = []  # this is not possible to exist for continuous action as there are unlimited number of action exist
        all_log_pis = []
        all_pol_regs = []
        all_baselines = []

        for a_i, pi, ob in zip(range(self.nagents), self.policies, obs):
            # curr_ac, probs, log_pi, pol_regs, ent = pi(ob, return_all_probs=True, return_log_pi=True, regularize=True, return_entropy=True)
            # curr_ac, log_pi, pol_regs, ent = pi(ob, return_all_probs=True, return_log_pi=True, regularize=True, return_entropy=True)
            curr_ac, log_pi, pol_regs = pi(ob, return_log_pi=True, regularize=True)
            # logger.add_scalar('agent%i/policy_entropy' % a_i, ent, self.niter)
            samp_acs.append(curr_ac)
            # all_probs.append(log_pi)
            all_log_pis.append(log_pi)
            all_pol_regs.append(pol_regs)

        critic_in = list(zip(obs, samp_acs))
        critic_rets = self.critic(critic_in)  # this is for current critic NN

        # ----------- baseline function for advantage function with continuous action space ---------
        # curT = time.time()
        # for a_i, pi, cur_obs in zip(range(self.nagents), self.policies, obs):
        #     sampled_Q = []
        #     act_clone = copy.deepcopy(acs)
        #     for _ in range(10):  # we sample an action for 100 times for individual agent
        #         sampled_action = pi(cur_obs).detach()
        #         act_clone[a_i] = sampled_action # replace the action from experience replay with the sampled_action from the policy.
        #         sample_critic_in = list(zip(obs, act_clone))
        #         sampled_Q_all = self.critic(sample_critic_in)
        #         sampled_Q.append(sampled_Q_all[a_i])
        #     baseline_expect_Q = torch.mean(torch.stack(sampled_Q))
        #     all_baselines.append(baseline_expect_Q)
        # endT = time.time()-curT
        # print("when sample 10 times, the time take is {} seconds".format(endT))
        # ----------- end of baseline function for advantage function with continuous action space ---------

        # for a_i, probs, log_pi, pol_regs, (q, all_q) in zip(range(self.nagents), all_probs, all_log_pis, all_pol_regs, critic_rets):
        for a_i, log_pi, pol_regs, q in zip(range(self.nagents), all_log_pis, all_pol_regs, critic_rets):
        # for a_i, log_pi, q, v in zip(range(self.nagents), all_log_pis, critic_rets, all_baselines):
            curr_agent = self.agents[a_i]
            # v = (all_q * probs).sum(dim=1, keepdim=True)  # this is the baseline function, or the "b"
            # pol_target = q - v
            pol_target = q
            if soft:
                pol_loss = (log_pi * (log_pi / self.reward_scale - pol_target).detach()).mean()
            else:
                pol_loss = (log_pi * (-pol_target).detach()).mean()

            # Create the dummy tensor with requires_grad=True
            # N = len(pol_target)
            # dummy_tensor = torch.ones((N, 1), requires_grad=True)
            # pol_loss = (dummy_tensor*(-pol_target).detach()).mean()

            for reg in pol_regs:
                pol_loss += 1e-3 * reg  # policy regularization

        # for a_i, q in zip(range(self.nagents), critic_rets):
        #     curr_agent = self.agents[a_i]
        #     pol_loss = -q.detach().mean()

            # don't want critic to accumulate gradients from policy loss
            disable_gradients(self.critic)
            pol_loss.backward()
            enable_gradients(self.critic)

            # grad_norm = torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()
            curr_agent.policy_optimizer.zero_grad()

            # if logger is not None:
            #     logger.add_scalar('agent%i/losses/pol_loss' % a_i,
            #                       pol_loss, self.niter)
            #     logger.add_scalar('agent%i/grad_norms/pi' % a_i,
            #                       grad_norm, self.niter)


    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.target_critic, self.critic, self.tau)
        for a in self.agents:
            soft_update(a.target_policy, a.policy, self.tau)

    def prep_training(self, device='gpu'):
        self.critic.train()
        self.target_critic.train()
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device=device)  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents],
                     'critic_params': {'critic': self.critic.state_dict(),
                                       'target_critic': self.target_critic.state_dict(),
                                       'critic_optimizer': self.critic_optimizer.state_dict()}}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, gamma=0.95, tau=0.01,
                      pi_lr=0.01, q_lr=0.01,
                      reward_scale=10.,
                      pol_hidden_dim=128, critic_hidden_dim=128, attend_heads=4,
                      **kwargs):
        """
        Instantiate instance of this class from multi-agent environment

        env: Multi-agent Gym environment
        gamma: discount factor
        tau: rate of update for target networks
        lr: learning rate for networks
        hidden_dim: number of hidden dimensions for networks
        """
        agent_init_params = []
        sa_size = []
        for acsp, obsp in zip(env.action_space, env.observation_space):
            # agent_init_params.append({'num_in_pol': obsp.shape[0],'num_out_pol': acsp.n})
            agent_init_params.append({'num_in_pol': obsp.shape[0],'num_out_pol': 2})
            # sa_size.append((obsp.shape[0], acsp.n))
            sa_size.append((obsp.shape[0], 2))

        init_dict = {'gamma': gamma, 'tau': tau,
                     'pi_lr': pi_lr, 'q_lr': q_lr,
                     'reward_scale': reward_scale,
                     'pol_hidden_dim': pol_hidden_dim,
                     'critic_hidden_dim': critic_hidden_dim,
                     'attend_heads': attend_heads,
                     'agent_init_params': agent_init_params,
                     'sa_size': sa_size}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename, load_critic=False):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)

        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])
        return instance