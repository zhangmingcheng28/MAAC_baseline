import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain


class MaddpgCritic(nn.Module):
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=False, attend_heads=1):
        super(MaddpgCritic, self).__init__()
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.feature_critics = nn.ModuleList()
        self.combine_critics = nn.ModuleList()
        obs_dim = sa_sizes[0][0] * self.nagents
        act_dim = sa_sizes[0][1] * self.nagents
        for sdim, adim in sa_sizes:
            feature_encoder = nn.Sequential()
            combine_encoder = nn.Sequential()
            feature_encoder.add_module('fc1', nn.Linear(obs_dim, 1024))
            feature_encoder.add_module('fc1_ac', nn.ReLU())
            self.feature_critics.append(feature_encoder)
            combine_encoder.add_module('fc2', nn.Linear(1024 + act_dim, 512))
            combine_encoder.add_module('fc2_ac', nn.ReLU())
            combine_encoder.add_module('fc3', nn.Linear(512, 300))
            combine_encoder.add_module('fc3_ac', nn.ReLU())
            combine_encoder.add_module('fc4', nn.Linear(300, 1))
            combine_encoder.add_module('fc4_ac', nn.ReLU())
            self.combine_critics.append(combine_encoder)

    def forward(self, obs, acts, agent_idx):
        state_features = self.feature_critics[agent_idx](obs)
        fea_act_combined = torch.cat([state_features, acts], dim=1)
        result = self.combine_critics[agent_idx](fea_act_combined)
        return result


class AttentionCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=False, attend_heads=1):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(AttentionCritic, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()

        self.state_encoders = nn.ModuleList()
        # iterate over agents
        for sdim, adim in sa_sizes:
            idim = sdim + adim
            odim = adim
            encoder = nn.Sequential()
            if norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                            affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
                                                      hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            # critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))  # original discrete policy
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, 1))  # for continuous policy
            self.critics.append(critic)

            state_encoder = nn.Sequential()
            if norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                            sdim, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
                                                            hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

        attend_dim = hidden_dim // attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for i in range(attend_heads):
            self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim,
                                                                attend_dim),
                                                       nn.LeakyReLU()))

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.nagents)

    def forward(self, inps, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        if agents is None:
            agents = range(len(self.critic_encoders))
        states = [s for s, a in inps]
        actions = [a for s, a in inps]
        inps = [torch.cat((s, a), dim=1) for s, a in inps]  # combine raw state and action for each agent
        # extract state-action encoding for each agent
        sa_encodings = []
        for encoder, inp in zip(self.critic_encoders, inps):  # there are N-seperated critic_encoders.
            one_sa = encoder(inp)  # batch norm, linear(20, 128), leakyRelu()
            sa_encodings.append(one_sa)
        # sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]
        # extract state encoding for each agent that we're returning Q for
        s_encodings = []
        for a_i in agents:
            one_s = self.state_encoders[a_i](states[a_i])  # batch norm, linear(18, 128), leakyRelu()
            s_encodings.append(one_s)
        # s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]
        # extract keys for each head for each agent
        all_head_keys = []
        for k_ext in self.key_extractors:
            each_head_all_agent_key = []
            for enc in sa_encodings:
                one_agent_keys = k_ext(enc)
                each_head_all_agent_key.append(one_agent_keys)
            all_head_keys.append(each_head_all_agent_key)

        # all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]  # this is the original version. All values the same, thanks to same torch seed.

        # extract sa values for each head for each agent
        all_head_values = []
        for v_ext in self.value_extractors:
            each_head_all_agent_value = []
            for enc in sa_encodings:
                one_agent_value = v_ext(enc)
                each_head_all_agent_value.append(one_agent_value)
            all_head_values.append(each_head_all_agent_value)
        # all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]  # original

        # extract selectors for each head for each agent that we're returning Q for
        all_head_selectors = []  # this is the queries vector
        for sel_ext in self.selector_extractors:
            each_head_selector = []
            for i, enc in enumerate(s_encodings):
                if i in agents:
                    one_agent_selector = sel_ext(enc)
                    each_head_selector.append(one_agent_selector)
            all_head_selectors.append(each_head_selector)

        # all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]for sel_ext in self.selector_extractors]  # original

        other_all_values = [[] for _ in range(len(agents))]
        all_attend_logits = [[] for _ in range(len(agents))]
        all_attend_probs = [[] for _ in range(len(agents))]
        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            # iterate over agents
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]
                values = [v for j, v in enumerate(curr_head_values) if j != a_i]
                # calculate attention across agents
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                attend_weights = F.softmax(scaled_attend_logits, dim=2)
                other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)
        # calculate Q per agent
        all_rets = []
        for i, a_i in enumerate(agents):
            head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
                               .mean()) for probs in all_attend_probs[i]]
            agent_rets = []
            # ------------- for continuous action -------------
            # For discrete action, we output Q-value for each possible action for each agent
            # But for continuous action, is not necessary, we only need to output a Q-value for the current SA pair
            # So, for continuous action, we just need Batch x 1 for the q-value
            critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1)
            q = self.critics[a_i](critic_in)
            # ------------ end for continuous action ---------
            # critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1)
            # all_q = self.critics[a_i](critic_in)
            # int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
            # q = all_q.gather(1, int_acs)
            if return_q:
                agent_rets.append(q)
            # if return_all_q:  # this return_all_q is only applicable for discrete action space
            #     agent_rets.append(all_q)
            if regularize:
                # regularize magnitude of attention logits
                attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                            all_attend_logits[i])
                regs = (attend_mag_reg,)
                agent_rets.append(regs)
            if return_attend:
                agent_rets.append(np.array(all_attend_probs[i]))
            if logger is not None:
                logger.add_scalars('agent%i/attention' % a_i,
                                   dict(('head%i_entropy' % h_i, ent) for h_i, ent
                                        in enumerate(head_entropies)),
                                   niter)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets
