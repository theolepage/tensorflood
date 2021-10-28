# TensorFlood

Tiny [automatic differentiation (autodiff)](https://en.wikipedia.org/wiki/Automatic_differentiation) engine implemented in Python. 

The following resources were used while creating this project:
- [Build Your Own Automatic Differentiation Program](https://towardsdatascience.com/build-your-own-automatic-differentiation-program-6ecd585eec2a) by Jonathan Kernes
- [Yaae](https://github.com/3outeille/Yaae) by Ferdinand Mom

## Example

The notebook `examples/classification.ipynb` demonstrates how to use the engine and how it can be used to train a small neural network.

<img src="./examples/classification.gif" width="50%">

## Usage

```python
from tensorflood.engine import Graph, Variable, Constant

with Graph() as g:
    x = Variable(2, name='x')
    y = Variable(10, name='y')
    c = Constant(7, name='c')
    
    z = (x * y) + c
    # z.data = 27
    
    z.backward()
    # x.grad = 10 (dz/dx)
    # y.grad = 2  (dz/dy)
    # c.grad = 1  (dz/dc)
```

## To-Do

- [ ] NN: Adam optimizer, more layers
- [ ] Merge `backward_` and `backward`
- [ ] Refactor `hasattr(node, 'input_nodes')`
- [ ] Unit tests with pytorch