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


@pytest.fixture
def six_walls():
    x0, y0, z0 = [random.random() for _ in range(3)]
    x1, y1, z1 = [2 + random.random() for _ in range(3)]
    x0_val, y0_val, z0_val = [FCNN(3, 1) for _ in range(3)]
    x1_val, y1_val, z1_val = [FCNN(3, 1) for _ in range(3)]
    x0_prime, y0_prime, z0_prime = [FCNN(3, 1) for _ in range(3)]
    x1_prime, y1_prime, z1_prime = [FCNN(3, 1) for _ in range(3)]

    EPS = 0.3
    g = Generator3D(
        xyz_min=(x0 + EPS, y0 + EPS, z0 + EPS),
        xyz_max=(x1 - EPS, y1 - EPS, z1 - EPS),
        method='equally-spaced',
    )
    g = StaticGenerator(g)
    g = SamplerGenerator(g)

    return (
        g,
        x0, y0, z0, x1, y1, z1,
        x0_val, y0_val, z0_val, x1_val, y1_val, z1_val,
        x0_prime, y0_prime, z0_prime, x1_prime, y1_prime, z1_prime,
    )


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


@pytest.mark.parametrize(argnames='coord_index', argvalues=[0, 1, 2])
def test_condition_3d_single_component(w0, f, g, xyz, net_31, coord_index):
    component = ConditionComponent3D(w0, f, g, coord_index=coord_index)
    condition = Condition3D(components=[component])
    xyz = list(xyz)
    xyz[coord_index] = torch.ones_like(xyz[coord_index], requires_grad=True) * (w0 + EPS)
    xyz_tensor = torch.cat(xyz, dim=1)

    u = condition.enforce(net_31, *xyz)
    assert all_close(u, f(xyz_tensor), atol=1e-4, rtol=1e-2)
    assert all_close(diff(u, xyz[coord_index]), g(xyz_tensor), atol=1e-4, rtol=1e-2)


def test_condition_3d_six_walls(six_walls, net_31):
    gen, x0, y0, z0, x1, y1, z1, x0_val, y0_val, z0_val, x1_val, y1_val, z1_val, \
    x0_prime, y0_prime, z0_prime, x1_prime, y1_prime, z1_prime = six_walls

    x, y, z = gen.get_examples()

    comp_x0 = ConditionComponent3D(x0, f_dirichlet=x0_val, f_neumann=x0_prime, coord_name='x')
    comp_y0 = ConditionComponent3D(y0, f_dirichlet=y0_val, f_neumann=y0_prime, coord_name='y')
    comp_z0 = ConditionComponent3D(z0, f_dirichlet=z0_val, f_neumann=z0_prime, coord_name='z')
    comp_x1 = ConditionComponent3D(x1, f_dirichlet=x1_val, f_neumann=x1_prime, coord_name='x')
    comp_y1 = ConditionComponent3D(y1, f_dirichlet=y1_val, f_neumann=y1_prime, coord_name='y')
    comp_z1 = ConditionComponent3D(z1, f_dirichlet=z1_val, f_neumann=z1_prime, coord_name='z')
    condition = Condition3D(components=[comp_x0, comp_y0, comp_z0, comp_x1, comp_y1, comp_z1])

    x0_tensor = (x0 + EPS) * torch.ones_like(x, requires_grad=True)
    u_x0 = condition.enforce(net_31, x0_tensor, y, z)
    assert all_close(u_x0, x0_val(torch.cat((x0_tensor, y, z), dim=1)))
    assert all_close(diff(u_x0, x0_tensor), x0_prime(torch.cat((x0_tensor, y, z), dim=1)), atol=1e-4, rtol=1e-2)

    x1_tensor = (x1 - EPS) * torch.ones_like(x, requires_grad=True)
    u_x1 = condition.enforce(net_31, x1_tensor, y, z)
    assert all_close(u_x1, x1_val(torch.cat((x1_tensor, y, z), dim=1)))
    assert all_close(diff(u_x1, x1_tensor), x1_prime(torch.cat((x1_tensor, y, z), dim=1)), atol=1e-4, rtol=1e-2)

    y0_tensor = (y0 + EPS) * torch.ones_like(y, requires_grad=True)
    u_y0 = condition.enforce(net_31, x, y0_tensor, z)
    assert all_close(u_y0, y0_val(torch.cat((x, y0_tensor, z), dim=1)))
    assert all_close(diff(u_y0, y0_tensor), y0_prime(torch.cat((x, y0_tensor, z), dim=1)), atol=1e-4, rtol=1e-2)

    y1_tensor = (y1 + EPS) * torch.ones_like(y, requires_grad=True)
    u_y1 = condition.enforce(net_31, x, y1_tensor, z)
    assert all_close(u_y1, y1_val(torch.cat((x, y1_tensor, z), dim=1)))
    assert all_close(diff(u_y1, y1_tensor), y1_prime(torch.cat((x, y1_tensor, z), dim=1)), atol=1e-4, rtol=1e-2)

    z0_tensor = (z0 + EPS) * torch.ones_like(z, requires_grad=True)
    u_z0 = condition.enforce(net_31, x, y, z0_tensor)
    assert all_close(u_z0, z0_val(torch.cat((x, y, z0_tensor), dim=1)))
    assert all_close(diff(u_z0, z0_tensor), z0_prime(torch.cat((x, y, z0_tensor), dim=1)), atol=1e-4, rtol=1e-2)

    z1_tensor = (z1 + EPS) * torch.ones_like(z, requires_grad=True)
    u_z1 = condition.enforce(net_31, x, y, z1_tensor)
    assert all_close(u_z1, z1_val(torch.cat((x, y, z1_tensor), dim=1)))
    assert all_close(diff(u_z1, z1_tensor), z1_prime(torch.cat((x, y, z1_tensor), dim=1)), atol=1e-4, rtol=1e-2)
