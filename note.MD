Reuse of activation layer leads to assertion error 

e.g.

```python
class MLP(nn.Module):
    """Multi Layer Perceptron with inplace option.
    Make sure inplace=true and false has the same visual graph"""

    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.activation = nn.ReLU(inplace)
        self.layers = nn.Sequential(
            nn.Linear(128, 128),
            self.activation,
            nn.Linear(128, 128),
            self.activation,
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x
```

x = get_module_summary(MLP(), torch.rand(2,128))
