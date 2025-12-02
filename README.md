# Universal PDE Solvers

Playground for universal PDE solvers.
We define a PDE as something in this form:
```
dudt = f[u, dudx, ddudxx, ...]
```
At the moment, all solvers follow a similar form:
1. Choose suitable timestep
2. Discretize the right-hand side, e.g. finite volume or finite differences
3. Integrate the time domain, e.g. a Runge-Kutta method
4. Apply boundary conditions

## Quickstart

```console
$ make
$ ./main 100
$ python3 ./plot.py
```
Note: Plotting requires the python libraries numpy, matplotlib, and scipy.

## Thirdparty

- [Igor](https://github.com/gidonbauer/Igor)
- Python: [Matplotlib](https://matplotlib.org/), [Numpy](https://numpy.org/), [Scipy](https://scipy.org/)