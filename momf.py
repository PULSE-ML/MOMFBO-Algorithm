from __future__ import annotations

from typing import Any, Callable, List, Optional, Union

import torch
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from torch import Tensor


class MOMF(qExpectedHypervolumeImprovement):
    def __init__(
        self,
        model: Model,
        ref_point: Union[List[float], Tensor],
        partitioning: NondominatedPartitioning,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
        X_pending: Optional[Tensor] = None,
        cost_call: Callable[Tensor, Tensor] = None,
        eta: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        r"""MOMF acquisition function supporting m>=2 outcomes.
        The model needs to have train_obj that has a fidelity
        objective appended to its end.
        In the following example we consider a 2-D output space
        but the ref_point is 3D because of fidelity objective.

        See [Irshad2021MOMF]_ for details.

        Example:
            >>> model = SingleTaskGP(train_X, train_Y)
            >>> ref_point = [0.0, 0.0, 0.0]
            >>> cost_func = lambda X:5+X[...,-1]
            >>> momf = MOMF(model, ref_point, partitioning,cost_func)
            >>> momf_val = momf(test_X)

        Args:
            model: A fitted model. There are two default assumptions in the training
                data. train_X should have fidelity parameter `s` as the last dimension
                of the input and train_Y contains a trust objective as its last
                dimension.
            ref_point: A list or tensor with `m+1` elements representing the reference
                point (in the outcome space) w.r.t. to which compute the hypervolume.
                The '+1' takes care of the trust objective appended to train_Y.
                This is a reference point for the objective values (i.e. after
                applying`objective` to the samples).
            partitioning: A `NondominatedPartitioning` module that provides the non-
                dominated front and a partitioning of the non-dominated space in hyper-
                rectangles. If constraints are present, this partitioning must only
                include feasible points.
            sampler: The sampler used to draw base samples. Defaults to
                `SobolQMCNormalSampler(num_samples=128, collapse_batch_dims=True)`.
            objective: The MCMultiOutputObjective under which the samples are evaluated.
                Defaults to `IdentityMultiOutputObjective()`.
            constraints: A list of callables, each mapping a Tensor of dimension
                `sample_shape x batch-shape x q x m` to a Tensor of dimension
                `sample_shape x batch-shape x q`, where negative values imply
                feasibility. The acqusition function will compute expected feasible
                hypervolume.
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation but have not yet
                been evaluated. Concatenated into `X` upon forward call. Copied and set
                to have no gradient.
            cost_call: A callable cost function mapping a Tensor of dimension
                `batch_shape x q x d` to a cost Tensor of dimension
                `batch_shape x q x m`.
                Defaults to an AffineCostModel with C(s)=1+s.
            eta: The temperature parameter for the sigmoid function used for the
                differentiable approximation of the constraints.
        """

        if len(ref_point) != partitioning.num_outcomes:
            raise ValueError(
                "The length of the reference point must match the number of outcomes. "
                f"Got ref_point with {len(ref_point)} elements, but expected "
                f"{partitioning.num_outcomes}."
            )
        ref_point = torch.as_tensor(
            ref_point,
            dtype=partitioning.pareto_Y.dtype,
            device=partitioning.pareto_Y.device,
        )
        super().__init__(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            X_pending=X_pending,
        )
        self.cost_call = cost_call

        if self.cost_call is None:
            cost_model = AffineFidelityCostModel(
                fidelity_weights={-1: 1.0}, fixed_cost=1.0
            )
        else:
            cost_model = GenericDeterministicModel(cost_call)
        cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
        self.cost_aware_utility = cost_aware_utility

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)
        hv_gain = self._compute_qehvi(samples=samples, X=X)
        cost_weighted_qehvi = self.cost_aware_utility(X=X, deltas=hv_gain)
        return cost_weighted_qehvi


import math

import torch
from botorch.test_functions.base import MultiObjectiveTestProblem
from torch import Tensor


class MOMFBraninCurrin(MultiObjectiveTestProblem):
    r"""Branin-Currin problem for multi-objective-multi-fidelity optimization.

    (2+1)-dimensional function with domain `[0,1]^3` where the last dimension
    is the fidelity parameter `s`.
    Both functions assume minimization. See [Irshad2021]_ for more details.

    Modified Branin function:

        B(x,s) = 21-((
        15*x_2 - b(s) * (15 * x_1 - 5) ** 2 + c(s) * (15 * x_1 - 5) - 6 ) ** 2
        + 10 * (1 - t(s)) * cos(15 * x_1 - 5)+10)/22

    Here `b`, `c`, `r` and `t` are constants and `s` is the fidelity parameter:
        where `b = 5.1 / (4 * math.pi ** 2) - 0.01(1-s)`,
        `c = 5 / math.pi - 0.1*(1 - s)`,
        `r = 6`,
        `t = 1 / (8 * math.pi) + 0.05*(1-s)`

    Modified Currin function:

        C(x) = 14-((1 - 0.1(1-s)exp(-1 / (2 * x_2))) * (
        2300 * x_1 ** 3 + 1900 * x_1 ** 2 + 2092 * x_1 + 60
        ) / 100 * x_1 ** 3 + 500 * x_1 ** 2 + 4 * x_2 + 20)/15

    """

    dim = 3
    num_objectives = 2
    _bounds = [(0.0, 1.0) for _ in range(dim)]
    _ref_point = [0, 0]
    _max_hv = 0.5235514158034145

    def _branin(self, X: Tensor) -> Tensor:
        x1 = X[..., 0]
        x2 = X[..., 1]
        s = X[..., 2]

        x11 = 15 * x1 - 5
        x22 = 15 * x2
        b = 5.1 / (4 * math.pi ** 2) - 0.01 * (1 - s)
        c = 5 / math.pi - 0.1 * (1 - s)
        r = 6
        t = 1 / (8 * math.pi) + 0.05 * (1 - s)
        y = (x22 - b * x11 ** 2 + c * x11 - r) ** 2 + 10 * (1 - t) * torch.cos(x11) + 10
        B = 21 - y
        return B / 22

    def _currin(self, X: Tensor) -> Tensor:
        x1 = X[..., 0]
        x2 = X[..., 1]
        s = X[..., 2]
        A = 2300 * x1 ** 3 + 1900 * x1 ** 2 + 2092 * x1 + 60
        B = 100 * x1 ** 3 + 500 * x1 ** 2 + 4 * x1 + 20
        y = (1 - 0.1 * (1 - s) * torch.exp(-1 / (2 * x2))) * A / B
        C = -y + 14
        return C / 15

    def evaluate_true(self, X: Tensor) -> Tensor:
        branin = self._branin(X)
        currin = self._currin(X)
        return torch.stack([-branin, -currin], dim=-1)


class MOMFPark(MultiObjectiveTestProblem):
    r"""Modified Park test functions for multi-objective multi-fidelity optimization.

    (4+1)-dimensional function with domain `[0,1]^5` where the last dimension
    is the fidelity parameter `s`. See [Irshad2021]_ for more details.

    The first modified Park function is

        P1(x, s)=A*(T1(x,s)+T2(x,s)-B)/22-0.8

    The second modified Park function is

        P2(x,s)=A*(5-2/3*exp(x1+x2)-x4*sin(x3)*A+x3-B)/4 - 0.7

    Here

        T_1(x,s) = (x1+0.001*(1-s))/2*sqrt(1+(x2+x3**2)*x4/(x1**2))

        T_2(x, s) = (x1+3*x4)*exp(1+sin(x3))

    and `A(s)=(0.9+0.1*s)`, `B(s)=0.1*(1-s)`.
    """

    dim = 5
    num_objectives = 2
    _bounds = [(0.0, 1.0) for _ in range(dim)]
    _ref_point = [0, 0]
    _max_hv = 0.08551927363087991

    def _transform(self, X: Tensor) -> Tensor:
        x1 = X[..., 0]
        x2 = X[..., 1]
        x3 = X[..., 2]
        x4 = X[..., 3]
        s = X[..., 4]
        _x1 = 1 - 2 * (x1 - 0.6) ** 2
        _x2 = x2
        _x3 = 1 - 3 * (x3 - 0.5) ** 2
        _x4 = 1 - (x4 - 0.8) ** 2
        return torch.stack([_x1, _x2, _x3, _x4, s],dim=-1)

    def _park1(self, X: Tensor) -> Tensor:
        x1 = X[..., 0]
        x2 = X[..., 1]
        x3 = X[..., 2]
        x4 = X[..., 3]
        s = X[..., 4]
        T1 = (
            (x1 + 1e-3 * (1 - s))
            / 2
            * torch.sqrt(1 + (x2 + x3 ** 2) * x4 / (x1 ** 2 + 1e-4))
        )
        T2 = (x1 + 3 * x4) * torch.exp(1 + torch.sin(x3))
        A = 0.9 + 0.1 * s
        B = 0.1 * (1 - s)
        return A * (T1 + T2 - B) / 22 - 0.8

    def _park2(self, X: Tensor) -> Tensor:
        x1 = X[..., 0]
        x2 = X[..., 1]
        x3 = X[..., 2]
        x4 = X[..., 3]
        s = X[..., 4]
        A = 0.9 + 0.1 * s
        B = 0.1 * (1 - s)
        return (
            A * (5 - 2 / 3 * torch.exp(x1 + x2) + x4 * torch.sin(x3) * A - x3 + B) / 4
            - 0.7
        )

    def evaluate_true(self, X: Tensor) -> Tensor:
        X = self._transform(X)
        park1 = self._park1(X)
        park2 = self._park2(X)
        return torch.stack([-park1, -park2], dim=-1)
