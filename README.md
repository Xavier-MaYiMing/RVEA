### RVEA: Reference vector-guided evolution algorithm

##### Reference: Cheng R, Jin Y, Olhofer M, et al. A reference vector guided evolutionary algorithm for many-objective optimization[J]. IEEE Transactions on Evolutionary Computation, 2016, 20(5): 773-791.

##### RVEA is a highly efficient many-objective optimization evolutionary algorithm (MaOEA). RVEA uses angle-penalized distance (APD) to rank the population in environmental selection.

| Variables | Meaning                                                      |
| --------- | ------------------------------------------------------------ |
| npop      | Population size                                              |
| iter      | Iteration number                                             |
| lb        | Lower bound                                                  |
| ub        | Upper bound                                                  |
| nobj      | The dimension of objective space                             |
| eta_c     | Spread factor distribution index (default = 30)              |
| eta_m     | Perturbance factor distribution index (default = 20)         |
| alpha     | The parameter to control the change rate of APD (default = 2) |
| fr        | Reference vector adaption parameter (default = 0.1)          |
| nvar      | The dimension of decision space                              |
| pop       | Population                                                   |
| objs      | Objectives                                                   |
| V0        | Original reference vectors                                   |
| V         | Reference vectors                                            |
| theta     | The smallest angle value of each reference vector to the others |
| APD       | Angle-penalized distance                                     |
| dom       | Domination matrix                                            |
| pf        | Pareto front                                                 |

#### Test problem: DTLZ1

$$
\begin{aligned}
	& k = nvar - nobj + 1, \text{ the last $k$ variables is represented as $x_M$} \\
	& g(x_M) = 100 \left[|x_M| + \sum_{x_i \in x_M}(x_i - 0.5)^2 - \cos(20\pi(x_i - 0.5)) \right] \\
	& \min \\
	& f_1(x) = \frac{1}{2}x_1x_2 \cdots x_{M - 1}(1 + g(x_M)) \\
	& f_2(x) = \frac{1}{2}x_1x_2 \cdots (1 - x_{M - 1})(1 + g(x_M)) \\
	& \vdots \\
	& f_{M - 1}(x) = \frac{1}{2}x_1(1 - x_2)(1 + g(x_M)) \\
	& f_M(x) = \frac{1}{2}(1 - x_1)(1 + g(x_M)) \\
	& \text{subject to} \\
	& x_i \in [0, 1], \quad \forall i = 1, \cdots, n
\end{aligned}
$$



#### Example

```python
if __name__ == '__main__':
    main(105, 500, np.array([0] * 7), np.array([1] * 7), 3)
```

##### Output:

![](https://github.com/Xavier-MaYiMing/RVEA/blob/main/Pareto%20front.png)



