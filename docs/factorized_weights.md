# Factorized Weight Initialization

## Overview

Added two new functions to `balloonpinnlib.py` for factorized weight initialization where **W = diag(S) × V**:
- **S**: Learnable diagonal scaling matrix (only diagonal elements are parameters)
- **V**: Fixed base weight matrix initialized using standard methods

## Functions

### 1. `init_factorized_weights(layer, init_method, s_init_value, dtype, device)`

Initialize a single `nn.Linear` layer with factorized weights.

**Parameters:**
- `layer` (nn.Linear): The linear layer to initialize
- `init_method` (callable): Initialization function for V (e.g., `nn.init.xavier_normal_`, `nn.init.kaiming_normal_`)
- `s_init_value` (float): Initial value for diagonal elements of S (default: 1.0)
- `dtype` (torch.dtype): Data type (default: torch.float32)
- `device` (torch.device): Device for parameters (default: None, uses layer's device)

**Returns:**
- `(S, V)`: Tuple of diagonal scaling vector and base weight matrix

**Example:**
```python
import torch.nn as nn
from balloonlib import balloonpinnlib as bpl

layer = nn.Linear(10, 5)
S, V = bpl.init_factorized_weights(
    layer,
    init_method=nn.init.xavier_normal_,
    s_init_value=1.0
)

# Now:
# - layer.S is a learnable parameter (shape: [5])
# - layer.V is a fixed buffer (shape: [5, 10])
# - Effective weight W = diag(S) × V is computed in forward pass
```

---

### 2. `init_factorized_network(module, init_method, s_init_value, dtype, exclude_layers)`

Apply factorized initialization to all `nn.Linear` layers in a module.

**Parameters:**
- `module` (nn.Module): Network to initialize
- `init_method` (callable): Initialization function for V matrices
- `s_init_value` (float): Initial value for S (default: 1.0)
- `dtype` (torch.dtype): Data type (default: torch.float32)
- `exclude_layers` (list): Layer names to exclude (default: None)

**Returns:**
- `dict`: Mapping of layer names to `(S, V)` tuples

**Example:**
```python
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

initialized = bpl.init_factorized_network(
    model,
    init_method=nn.init.kaiming_normal_,
    s_init_value=1.0
)

# All Linear layers now use factorized weights
# S parameters will be optimized during training
```

---

## How It Works

### Mathematical Formulation

Instead of learning the full weight matrix **W** directly, we factorize it as:

```
W = diag(S) × V
```

Where:
- **S** ∈ ℝⁿ: A vector representing the diagonal of a scaling matrix
- **V** ∈ ℝⁿˣᵐ: A fixed base weight matrix
- Element-wise: `W[i,j] = S[i] × V[i,j]`

### Implementation Details

1. **Initialization:**
   - V is initialized using the provided method (e.g., Xavier, Kaiming)
   - S is initialized to `s_init_value` (typically 1.0 for identity scaling)

2. **Storage:**
   - V is stored as a **buffer** (not optimized)
   - S is stored as a **parameter** (learnable)

3. **Forward Pass:**
   ```python
   def forward(x):
       W = self.S.unsqueeze(1) * self.V  # Compute W = diag(S) × V
       return F.linear(x, W, self.bias)
   ```

4. **Training:**
   - Only S is updated by the optimizer
   - V remains fixed throughout training
   - Reduces number of learnable parameters

---

## Benefits

1. **Parameter Efficiency**: Learning n diagonal elements instead of n×m full matrix elements
2. **Adaptive Scaling**: Each output neuron gets its own learnable scaling factor
3. **Stable Initialization**: V provides a good initialization that gets adaptively scaled
4. **Regularization**: Constrains the learning process, potentially improving generalization

---

## Usage in Multihead Class

To use in your existing `Multihead` class, modify the `init_nn_params` method:

```python
def init_nn_params(self):
    """Factorized weight initialization"""
    init_factorized_network(
        self,
        init_method=nn.init.xavier_normal_,
        s_init_value=1.0,
        exclude_layers=['some_layer_name']  # Optional
    )
```

Or for selective initialization:

```python
def init_nn_params(self):
    """Mixed initialization: factorized for some layers, standard for others"""
    # Factorize main network layers
    for layer in self.Sequential:
        if isinstance(layer, nn.Linear):
            init_factorized_weights(layer, nn.init.xavier_normal_)
    
    # Standard initialization for output layers
    for layer in self.nv_final_layers:
        nn.init.xavier_normal_(layer.weight)
```

---

## Supported Initialization Methods

Any PyTorch initialization function that modifies a tensor in-place:

- `nn.init.xavier_normal_`
- `nn.init.xavier_uniform_`
- `nn.init.kaiming_normal_`
- `nn.init.kaiming_uniform_`
- `nn.init.orthogonal_`
- `nn.init.normal_`
- `nn.init.uniform_`
- Custom functions with signature: `fn(tensor) -> tensor`

---

## Training Considerations

**Optimizer Setup:**
```python
# S parameters will automatically be included
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

**Monitoring S Values:**
```python
for name, param in model.named_parameters():
    if 'S' in name:
        print(f"{name}: {param.data}")
```

**Accessing Effective Weights:**
```python
# For a factorized layer:
W_effective = layer.S.unsqueeze(1) * layer.V
```

---

## Notes

- The factorization is transparent to the forward pass
- Gradients flow correctly through both S and the computation
- Compatible with all PyTorch optimizers
- Can be combined with other regularization techniques
- V is fixed after initialization, only S is learnable
