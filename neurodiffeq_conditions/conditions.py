import torch
from neurodiffeq.conditions import BaseCondition
from neurodiffeq import diff


class ConditionComponent3D:
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

    _index_lookup = dict(x=0, y=1, z=2)

    def __init__(self, w, f_dirichlet=None, f_neumann=None, coord_index=None, coord_name=None):
        if not (f_dirichlet or f_neumann):
            raise ValueError("Either `f_dirichlet` or `f_neumann` must be specified")
        self.w = w
        self.f_d, self.f_n = f_dirichlet, f_neumann
        self.idx = coord_index if (coord_index is not None) else self._index_lookup[coord_name]

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
            D = self.f_d(p_tensor) - net(p_tensor)

        if self.f_n is None:
            N = 0.0
        else:
            N = self.f_n(p_tensor) - diff(net(p_tensor), projection[self.idx])

        return D, N
