"""Modify standard PyTorch distributions so they to make compatible with this codebase."""
import math
import torch
import torch.nn as nn
from harl.utils.models_tools import init, get_init_method


class FixedCategorical(torch.distributions.Categorical):
    """Modify standard PyTorch Categorical."""

    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class FixedNormal(torch.distributions.Normal):
    """Modify standard PyTorch Normal."""

    def log_probs(self, actions):
        return super().log_prob(actions)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class FixedBeta(torch.distributions.Beta):
    """Unit-interval actions; deterministic evaluation uses the mean by design."""

    def log_probs(self, actions):
        return super().log_prob(actions)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class FixedSqrtBeta(FixedBeta):
    """X is a Beta release fraction; U=sqrt(X) enters the quadratic interface."""

    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape).sqrt()

    def log_prob(self, value):
        # p_U(u)=p_X(u^2)*2u. Never attach a Beta(X) density directly to U.
        if self._validate_args:
            self._validate_sample(value)
        return super().log_prob(value.square()) + torch.log(2.*value)

    def log_probs(self, actions):
        return self.log_prob(actions)

    def entropy(self):
        expected_log_x = torch.digamma(self.concentration1)-torch.digamma(
            self.concentration1+self.concentration0)
        return (torch.distributions.Beta.entropy(self)-math.log(2.)-.5*expected_log_x).sum(-1)

    @property
    def mean(self):
        a, b = self.concentration1, self.concentration0
        return (torch.lgamma(a+.5)+torch.lgamma(a+b)-torch.lgamma(a)-torch.lgamma(a+b+.5)).exp()

    @property
    def variance(self):
        return (self.concentration1/(self.concentration1+self.concentration0)-self.mean.square()).clamp_min(0.)

    def mode(self):
        # Deployment selects E[X], hence Q_det=E[Q] at fixed feasible bounds.
        return (self.concentration1/(self.concentration1+self.concentration0)).sqrt()


class BetaHead(nn.Module):
    """Optional bounded head, leaving the existing Gaussian default unchanged."""

    def __init__(self, num_inputs, num_outputs, initialization_method, gain, concentration_offset=0., action_coordinate='control'):
        super().__init__()
        if action_coordinate not in ('control', 'release_fraction'):
            raise ValueError('Beta action coordinate must be control or release_fraction')
        self.action_coordinate = action_coordinate
        self.concentration_offset = float(concentration_offset)
        if not math.isfinite(self.concentration_offset) or self.concentration_offset < 0:
            raise ValueError("Beta concentration offset must be finite and nonnegative")
        self.linear = init(
            nn.Linear(num_inputs, 2 * num_outputs),
            get_init_method(initialization_method),
            lambda x: nn.init.constant_(x, 0),
            gain,
        )
        if action_coordinate == 'release_fraction':
            # Preserve the initial quarter-interval deterministic release at
            # zero features: sqrt(alpha/(alpha+beta))=.5, just as before.
            alpha = 1.+math.log(2.)
            with torch.no_grad():
                self.linear.bias[num_outputs:].fill_(math.log(math.expm1(3.*alpha-1.)))

    def forward(self, x, available_actions=None):
        # Shapes >= 1 avoid singular boundary densities. Keep PyTorch's native
        # Beta sample, rather than clipping samples into artificial point masses.
        shapes = torch.nn.functional.softplus(self.linear(x)) + 1.0
        alpha, beta = shapes.chunk(2, dim=-1)
        if self.concentration_offset:
            # Preserve the original mean function for the same network weights.
            # Only concentration changes: S_new = S_old + offset.
            mean = alpha / (alpha + beta)
            alpha = alpha + self.concentration_offset * mean
            beta = beta + self.concentration_offset * (1. - mean)
        distribution = FixedSqrtBeta if self.action_coordinate == 'release_fraction' else FixedBeta
        return distribution(alpha, beta)


class Categorical(nn.Module):
    """A linear layer followed by a Categorical distribution."""

    def __init__(
        self, num_inputs, num_outputs, initialization_method="orthogonal_", gain=0.01
    ):
        super(Categorical, self).__init__()
        init_method = get_init_method(initialization_method)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    """A linear layer followed by a Diagonal Gaussian distribution."""

    def __init__(
        self,
        num_inputs,
        num_outputs,
        initialization_method="orthogonal_",
        gain=0.01,
        args=None,
    ):
        super(DiagGaussian, self).__init__()

        init_method = get_init_method(initialization_method)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        if args is not None:
            self.std_x_coef = args["std_x_coef"]
            self.std_y_coef = args["std_y_coef"]
        else:
            self.std_x_coef = 1.0
            self.std_y_coef = 0.5
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        log_std = torch.ones(num_outputs) * self.std_x_coef
        self.log_std = torch.nn.Parameter(log_std)

    def forward(self, x, available_actions=None):
        action_mean = self.fc_mean(x)
        action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef
        return FixedNormal(action_mean, action_std)
