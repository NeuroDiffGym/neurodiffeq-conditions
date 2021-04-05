import random
import pytest

import torch

from neurodiffeq import diff
from neurodiffeq.networks import FCNN
from neurodiffeq.generators import Generator3D, SamplerGenerator, StaticGenerator
from neurodiffeq_conditions.conditions import ConditionComponent3D, Condition3D

N = 10
EPS = 1e-10


@pytest.fixture
def w0():
    return random.random()


@pytest.fixture
def f():
    return FCNN(3, 1, hidden_units=(2,))


@pytest.fixture
def g():
    return FCNN(3, 1, hidden_units=(2,))


@pytest.fixture
def xyz():
    return tuple(torch.rand(N, 1, requires_grad=True) for _ in range(3))


@pytest.fixture
def net_31():
    return FCNN(3, 1)


def lookup_index(coord_name):
    idx = dict(x=0, y=1, z=2).get(coord_name)
    if idx is None:
        raise RuntimeError('Unknown coord_name =', coord_name)
    return idx


def all_close(x, y, **kwargs):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)

    # manage shape
    xy = x + y
    x = xy - y
    y = xy - x

    return torch.isclose(x, y, **kwargs).all()


@pytest.mark.parametrize(
    argnames=['coord_index', 'coord_name'],
    argvalues=[[0, 'x'], [1, 'y'], [2, 'z'], [1, 'x']]
)
def test_condition_component_3d_get_projection(w0, f, g, xyz, coord_index, coord_name):
    component = ConditionComponent3D(w0, f_dirichlet=f, f_neumann=g, coord_index=coord_index)
    projection = component._get_projection(*xyz)
    assert all_close(w0, projection[coord_index])

    component = ConditionComponent3D(w0, f_dirichlet=f, f_neumann=g, coord_name=coord_name)
    new_coord_index = lookup_index(coord_name)
    projection = component._get_projection(*xyz)
    assert all_close(w0, projection[new_coord_index])

    component = ConditionComponent3D(w0, f_dirichlet=f, f_neumann=g, coord_index=coord_index, coord_name=coord_name)
    projection = component._get_projection(*xyz)
    assert all_close(w0, projection[coord_index])


@pytest.mark.parametrize(argnames='coord_index', argvalues=[0, 1, 2])
def test_condition_component_3d_signed_distance_from(w0, f, g, xyz, coord_index):
    component = ConditionComponent3D(w0, f, g, coord_index=coord_index)
    d = component.signed_distance_from(*xyz)
    assert all_close(d, xyz[coord_index] - w0)


@pytest.mark.parametrize(argnames='coord_index', argvalues=[0, 1, 2])
def test_condition_component_3d_get_dn(w0, f, g, xyz, net_31, coord_index):
    component = ConditionComponent3D(w0, f, g, coord_index=coord_index)
    p = component._get_projection(*xyz)
    D, N = component.get_dn(net_31, *xyz)
    p_tensor = torch.cat(p, dim=1)
    assert all_close(D, f(p_tensor) - net_31(p_tensor))
    assert all_close(N, g(p_tensor) - diff(net_31(p_tensor), p[coord_index]))

    component = ConditionComponent3D(w0, None, g, coord_index=coord_index)
    p = component._get_projection(*xyz)
    D, N = component.get_dn(net_31, *xyz)
    p_tensor = torch.cat(p, dim=1)
    assert all_close(D, 0.0)
    assert all_close(N, g(p_tensor) - diff(net_31(p_tensor), p[coord_index]))

    component = ConditionComponent3D(w0, f, None, coord_index=coord_index)
    p = component._get_projection(*xyz)
    D, N = component.get_dn(net_31, *xyz)
    p_tensor = torch.cat(p, dim=1)
    assert all_close(D, f(p_tensor) - net_31(p_tensor))
    assert all_close(N, 0.0)


