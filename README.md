# Neurodiffeq Conditions

This library extends the [neurodiffeq](https://github.com/NeuroDiffGym/neurodiffeq) library by providing more initial / boundary conditions. 

**The APIs are NOT finalized until incorporated into the main neurodiffeq library. USE WITH CAUTION.**

# Install

**To simply install/update to latest version**

```bash
pip install -U git+https://github.com/NeuroDiffGym/neurodiffeq-conditions
```

**Alternatively, to edit and develop this library**

```bash
git clone https://github.com/NeuroDiffGym/neurodiffeq-conditions
cd neurodiffeq-conditions && pip install -e .
git checkout <COMMIT-OR-BRANCH-OR-TAG>
```

# Conditions

### Dirichlet-Neumann Mixed 3D

A boundary condition imposed on a rectangular region in R^3. To construct such a condition, first decompose the condition into several components.

**Imposing condition on a 2-D plane perpendicular to x-axis**

```python
import torch
from neurodiffeq_conditions import ConditionComponent3D

# Just Dirichlet: u(x=0.0, y, z) = f(y,z) = sin(y) + cos(z) 
f = lambda x, y, z: torch.sin(y) + torch.cos(z)
comp_x0 = ConditionComponent3D(0.0, f_dirichlet=f, coord_name='x')

# Just Neumann: u'_x(x=1.0, y, z) = g(y,z) = 2*y + 3*z
g = lambda x, y, z: 2*x + 3*z
comp_x1 = ConditionComponent3D(1.0, f_neumann=g, coord_name='x')

# Dirichlet: u(x=2.0, y, z) = f(y,z) = sin(y) + cos(z) 
# Neumann: u'_x(x=2.0, y, z) = g(y,z) = 2*y + 3*z
f = lambda x, y, z: torch.sin(y) + torch.cos(z)
g = lambda x, y, z: 2*x + 3*z
comp_x2 = ConditionComponent3D(2.0, f_dirichlet=f, f_neumann=g, coord_name='x')
```

**Imposing condition on a 2-D plane perpendicular to y- or z-axis**

```python
# similar to above, just change `coord_name='x'` to `coord_name='y'`
comp_y0 = ConditionComponent3D(..., coord_name='y')
comp_y1 = ConditionComponent3D(..., coord_name='y')
# or change it to 'z'
comp_z0 = ConditionComponent3D(..., coord_name='z')
comp_z1 = ConditionComponent3D(..., coord_name='z')
```

**Putting Them Together** 

```python
from neurodiffeq_conditions import Condition3D

# You can put in arbitraily many components, in any order
condition = Condition3D(components=[comp_x0, comp_x1, comp_x2, comp_y0, comp_y1, comp_z0, comp_z1, ...])
```

The constructed `condition` can now be used for neurodiffeq.

*Note: The condition depends on limits when evaluating on points **exactly on** the boudary. Since PyTorch is a numerical package and cannot take limits, you need to make sure that points **on** the boundary are never sampled. Use points close to boundaries instead. Say, add an EPS=10^-6 to points exactly on the boundary.* 
