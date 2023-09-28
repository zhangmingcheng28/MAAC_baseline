import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
from utils.misc import onehot_from_logits, categorical_sample

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class BasePolicy(nn.Module):
    """
    Base policy network
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.leaky_relu,
                 norm_in=True, onehot_dim=0):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(BasePolicy, self).__init__()

        init_w = 3e-3

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim, affine=False)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim + onehot_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, out_dim)  # this is the original
        # self.fc3_cont = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.Tanh())
        self.nonlin = nonlin
        # --------- for continuous policy ------------------
        self.log_std_min = -20
        self.log_std_max = 2
        self.mean_linear = nn.Linear(hidden_dim, out_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        # parameterizing in terms of log of standard deviation offers numerical stability,
        # ensures positive standard deviations, and can make optimization more effective and robust.
        self.log_std_linear = nn.Linear(hidden_dim, out_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        # --------- end for continuous policy ------------------



    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations (optionally a tuple that
                                additionally includes a onehot label)
        Outputs:
            out (PyTorch Matrix): Actions
        """
        onehot = None
        if type(X) is tuple:
            X, onehot = X
        inp = self.in_fn(X)  # don't batchnorm onehot
        if onehot is not None:
            inp = torch.cat((onehot, inp), dim=1)
        h1 = self.nonlin(self.fc1(inp))
        h2 = self.nonlin(self.fc2(h1))
        # out = self.fc3(h2)  # this is the original for discrete action space
        # out = self.fc3_cont(h2)  # this is the original for discrete action space
        # ---------- for continuous action space ----------
        mean = self.mean_linear(h2)
        log_std = self.log_std_linear(h2)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        # out = (mean, log_std)
        # ---------- end for continuous action space ----------
        return mean, log_std
        # return out


class DiscretePolicy(BasePolicy):
    """
    Policy Network for discrete action spaces
    """
    def __init__(self, *args, **kwargs):
        super(DiscretePolicy, self).__init__(*args, **kwargs)

    def forward(self, obs, sample=True, return_all_probs=False,
                return_log_pi=False, regularize=False,
                return_entropy=False):
        # out = super(DiscretePolicy, self).forward(obs)  # for discrete, original
        mean, log_std = super(DiscretePolicy, self).forward(obs)
        # ----- for continuous policy --------
        std = log_std.exp()
        normal = Normal(0, 1)  # this is the re-parameterization trick
        z = normal.sample(mean.shape)
        action = torch.tanh(mean + std * z.to(device))
        rets = [action]
        # # rets = [out]
        if return_log_pi:
            log_prob = Normal(mean, std).log_prob(mean + std*z.to(device)) - torch.log(1 - action.pow(2) + 1e-6)
            # log_prob = Normal(mean, std).log_prob(mean + std*z.to(device)) - torch.log(1 - action.pow(2) + 1e-6) - torch.log(torch.tensor(5))
            log_prob = log_prob.sum(dim=-1, keepdim=True)  # required for multi-dimensional continuous action space, sum action prob across entire action vector
            rets.append(log_prob)
        if regularize:
            rets.append([(mean**2).mean() + (log_std**2).mean()])
        if return_entropy:
            entropy = 0.5 + 0.5 * log_std.sum(dim=-1, keepdim=True) + 0.5 * mean.shape[-1] * math.log(2 * math.pi)
            rets.append(entropy.mean())
        if len(rets) == 1:
            return rets[0]
        return rets
        # return out
        # ----- end for continuous policy --------

        # probs = F.softmax(out, dim=1)
        # on_gpu = next(self.parameters()).is_cuda
        # if sample:
        #     int_act, act = categorical_sample(probs, use_cuda=on_gpu)
        # else:
        #     act = onehot_from_logits(probs)
        # rets = [act]
        # if return_log_pi or return_entropy:
        #     # to compute the log probabilities of EACH action, it is only required in discrete action space
        #     log_probs = F.log_softmax(out, dim=1)
        # if return_all_probs:
        #     rets.append(probs)
        # if return_log_pi:
        #     # return log probability of SELECTED action
        #     rets.append(log_probs.gather(1, int_act))
        # if regularize:
        #     rets.append([(out**2).mean()])
        # if return_entropy:
        #     rets.append(-(log_probs * probs).sum(1).mean())
        # if len(rets) == 1:
        #     return rets[0]
        # return rets
