import torch
from neurodiffeq.conditions import BaseCondition
from neurodiffeq import diff
from neurodiffeq._version_utils import warn_deprecate_class
from abc import ABC, abstractmethod


class ComposedCondition(BaseCondition):
    r"""A Neumann-Dirichlet mixed condition defined on a N-dimensional region.
    The conditions are enforced on (N-1)-dimensional hyperplanes perpendicular to one axis.
    :param: Components of this condition.
    :type: list[ConditionComponent]
    """

    def __init__(self, components=None, detach=False, k=2):
        super().__init__()
        if k <= 1:
            raise ValueError("k must be greater than 1")
        self.k = k
        # make sure components is ordered
        self.components = list(components or [])
        self.detach = detach

    def enforce(self, net, *coords):
        if not self.components:
            return net(torch.cat(coords, dim=1))

        DNs = tuple(comp.get_dn(net, *coords) for comp in self.components)
        ls = tuple(comp.signed_distance_from(*coords) for comp in self.components)
        if self.detach:
            ws = [l.detach() ** self.k for l in ls]
        else:
            ws = [l ** self.k for l in ls]
        numerator = sum(map(
            lambda DN_l_w: (DN_l_w[0][0] + DN_l_w[0][1] * DN_l_w[1]) / DN_l_w[2],
            zip(DNs, ls, ws)
        ))
        denominator = sum(map(lambda w: 1 / w, ws))
        return net(torch.cat(coords, dim=1)) + numerator / denominator


class ComposedCondition3D(ComposedCondition):
    r"""A Neumann-Dirichlet mixed condition defined on a 3D region.
    The conditions are enforced on 2-D planes perpendicular to :math:`x`-, :math:`y`-, or :math:`z`-axis.
    :param: Components of this condition.
    :type: list[ConditionComponent3D]
    """

    def enforce(self, net, x, y, z):
        return super().enforce(net, x, y, z)


class ComposedCondition2D(ComposedCondition):
    def enforce(self, net, x, y):
        return super().enforce(net, x, y)


class BaseConditionComponent(ABC):
    @abstractmethod
    def signed_distance_from(self, *x):
        pass

    @abstractmethod
    def get_dn(self, *x):
        pass


class ConditionComponent(BaseConditionComponent):
    r"""Component of a N-dimensional BC/IC condition. Must be used together with a ComposedCondition.
    This component enforces

    - Dirichlet condition :math:`u(x_0, x_1, \dots, x_{n-1})= f_d(x_0, x_1, \dots, x_{n-1})`
      on :math:`x_i = x_i^*` where :math:`i \in \{0, ..., n-1\}`; and/or
    - Neumann condition :math:`\frac{\partial}{\partial x_i} u(x_0, x_1, \dots, x_{n-1})= f_n(x_0, x_1, \dots, x_{n-1})`
      on :math:`x_i = x_i^*` where :math:`i \in \{0, ..., n-1\}`

    :param w: Value of :math:`x_i^*`
    :type w: float
    :param f_dirichlet: Dirichlet condition at :math:`x_i = x_i^*`, mapping every point on boundary to a value.
    :type f_dirichlet: callable, optional
    :param f_neumann: Neumann condition at :math:`x_i = x_i^*`, mapping every point on boundary to a value.
    :type f_neumann: callable, optional
    :param coord_index: Index :math:`i` of :math:`x_i`.
    :type coord_index: int
    """

    _index_lookup = {}

    def __init__(self, w, f_dirichlet=None, f_neumann=None, coord_index=None, coord_name=None):
        if not (f_dirichlet or f_neumann):
            raise ValueError("Either `f_dirichlet` or `f_neumann` must be specified")
        self.w = w
        self.f_d, self.f_n = f_dirichlet, f_neumann
        if coord_index is None:
            self.idx = self._index_lookup[coord_name]
        else:
            self.idx = coord_index

    def signed_distance_from(self, *x):
        return x[self.idx] - self.w

    def _get_projection(self, *x):
        l = list(x)
        l[self.idx] = self.w * torch.ones_like(l[self.idx], requires_grad=(self.f_n is not None))
        return l

    def get_dn(self, net, *x):
        projection = self._get_projection(*x)
        p_tensor = torch.cat(projection, dim=1)

        if self.f_d is None:
            D = 0.0
        else:
            D = self.f_d(*projection) - net(p_tensor)

        if self.f_n is None:
            N = 0.0
        else:
            N = self.f_n(*projection) - diff(net(p_tensor), projection[self.idx])

        return D, N


class RobinConditionComponent(ConditionComponent):
    _index_lookup = {}

    def __init__(self, w, a, b, f, coord_index=None, coord_name=None):
        self.w = w
        self.a = a
        self.b = b
        self.f = f
        if coord_index is None:
            self.idx = self._index_lookup[coord_name]
        else:
            self.idx = coord_index

    def _get_projection(self, *x):
        l = list(x)
        l[self.idx] = self.w * torch.ones_like(l[self.idx], requires_grad=True)
        return l

    def get_dn(self, net, *x):
        projection = self._get_projection(*x)
        p_tensor = torch.cat(projection, dim=1)

        output_val = net(p_tensor)
        normal_der = diff(output_val, projection[self.idx])
        R = (self.f(*projection) - self.a * output_val - self.b * normal_der) / 2

        return R / self.a, R / self.b


class ConditionComponent3D(ConditionComponent):
    r"""Component of a three-dimensional BC/IC condition. Must be used together with a Condition3D.
    This component enforces

    - Dirichlet condition :math:`u(x, y, z, \dots)= f_d(x, y, z, \dots)`
      on :math:`w = w_0` where :math:`w` is either :math:`x`, :math:`y`, or :math:`z`; and/or
    - Neumann condition :math:`\nabla_n u(x, y, z, \dots)= f_n(x, y, z, \dots)`
      on :math:`w = w_0` where :math:`w` is either :math:`x`, :math:`y`, or :math:`z`.

    :param w: Value of :math:`x_0`, :math:`y_0`, or :math:`z_0`.
    :type w: float
    :param f_dirichlet: Dirichlet condition at :math:`w = w_0`, mapping point (x, y, z) on boundary to a value.
    :type f_dirichlet: callable, optional
    :param f_neumann: Neumann condition at :math:`w = w_0`, mapping point (x, y, z) on boundary to a value.
    :type f_neumann: callable, optional
    :param coord_index: Index of `w`. 0 for 'x', 1 for 'y', 2 for 'z'. If provided, `coord_name` will be ignored.
    :type coord_index: int
    :param coord_name: Name of `w`. Either 'x', 'y', or 'z'.
    :type coord_name: str
    """

    _index_lookup = dict(x=0, y=1, z=2, t=2)


class ConditionComponent2D(ConditionComponent):
    _index_lookup = dict(x=0, y=1, t=1)


class BoundedComponent2D(ConditionComponent):
    _index_lookup = dict(x=0, y=1, t=1)

    def __init__(self, w, f_dirichlet=None, f_neumann=None, coord_index=None, coord_name=None,
                 v_lower=None, v_upper=None):
        super(BoundedComponent2D, self).__init__(
            w=w, f_dirichlet=f_dirichlet, f_neumann=f_neumann, coord_index=coord_index, coord_name=coord_name)
        if v_lower is not None and v_upper is not None and v_lower > v_upper:
            raise ValueError(f"v_lower={v_lower} cannot be greater than v_upper={v_upper}")
        self.v_lower = v_lower
        self.v_upper = v_upper

    def _get_projection(self, *x):
        if self.idx == 0:
            w, v = x
        else:
            v, w = x

        proj_w = self.w * torch.ones_like(w, requires_grad=(self.f_n is not None))
        proj_v = torch.clone(v)
        if self.v_upper is not None:
            proj_v[v > self.v_upper] = self.v_upper
        if self.v_lower is not None:
            proj_v[v < self.v_lower] = self.v_lower

        if self.idx == 0:
            return proj_w, proj_v
        else:
            return proj_v, proj_w


# DEPRECATED NAMES
Condition3D = warn_deprecate_class(ComposedCondition3D)
