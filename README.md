# HQCMCBD (Hybrid Quantum-Classical Multi-cuts Benders' Decomposition) Algorithm

<br>
**HQCMCBD** (Hybrid Quantum-Classical Multi-cuts Benders' Decomposition) Algorithm is a software package for Mixed-integer Linear Programming (MILP) optimization.

HQCMCBD emploies both direct and hybrid quantum optimization models named [Binary Quadratic Models (BQM)](https://docs.ocean.dwavesys.com/en/stable/concepts/bqm.html) and [Constrained Quadratic Models (CQM)](https://docs.ocean.dwavesys.com/en/stable/concepts/cqm.html) on D-Wave quantum computers ([D-Wave systems](https://www.dwavesys.com/)). In general, HQCMCBD is a hybrid quantum-classical version of the traditional Benders Decomposition (BD) (See [Geoffrion AM (1972)](https://link.springer.com/article/10.1007/bf00934810), [Van Slyke, R. M (1969)](https://epubs.siam.org/doi/abs/10.1137/0117061)), Unlike the classical BD, HQCMCBD demonstrates a significant advantage in solving MILP optimization problems (Such as [[1]](https://ieeexplore.ieee.org/abstract/document/9771632) and [[2]](https://ieeexplore.ieee.org/abstract/document/10232912)).

## Why HQCMCBD?

The strength of HQCMCBD lies in its ability to leverage both quantum and classical computing techniques to solve large-scale, mixed-integer linear programming problems, which are notoriously difficult to manage with traditional methods. By breaking the problem into smaller, more manageable sub-problems (cuts) and iteratively refining the solution, HQCMCBD accelerates the convergence towards an optimal solution. This hybrid approach also offers the flexibility of harnessing the power of quantum computing for specific sub-tasks while relying on classical methods for others, maximizing computational efficiency.

###### HQCMCBD is for everyone!
- Professionals pursuing an *off-the-shelf* MILP optimization solver to tackle problems in operations research (e.g., power systems, communication system, supply chains, manufacturing, health care, etc.),
- Researchers who hope to advance the theory and algorithms of optimization via quantum technologies,
- Experts in quantum computation who want to experiment with hyperparameters and/or encodings in QHD to achieve even better practical performance.

### Requirement
`Python >= 3.8`, `Gurobi>=9.1.2`, `dwave-system>=1.10.0`

## Usage

A good example notebooks for a jump start is `Example.ipynb`. The following illustrates the basic building blocks of QHDOPT and their functionalities briefly.

In `.py` Import HQCMCBD by running

```python
from HQCMCBD import HQCMCBD_algorithm
from HQCMCBD import *
```

In `.ipynb` Import HQCMCBD by running

```python
%run HQCMCBD_notebook.ipynb
```

You can create a problem instance by directly constructing the function via `Gurobi` first. Then, you can use our solver to get the solutions.

```python
# Optimize the model
model = gp.Model("Example_model")
...
model.optimize()
Solver = HQCMCBD_algorithm(model, mode = "manual")
Solver.run()
```

### Configuration and parameters
you need to setup the config before you run the solver. Here is an example configuration `json`

```JSON
{
    "lambda_var": {
        "nonneg_bits_length": 15,
        "decimal_bits_length": 4,
        "negative_bits_length": 15
    },
    "submethod": "l_shape",
    "debug_mode": true,
    "Hybrid_mode": false,
    "dwave": {
        "mode": "cqm",
        "DWave_token": "example-token",
        "Mcut_flag": false,
        "Cutnums": 1,
        "num_of_read": 100
    },
    "threshold":{
        "type": "absolute",
        "gap": 0.5
    },
    "max_steps": 20
}
```
| Keyword              | Meaning | Comment |
| :----------------     | :------: | :----: |
| `lambda_var`  |   Connnection varibale $$t$$ in [[1]](https://ieeexplore.ieee.org/abstract/document/9771632) and $$\lambda$$ in [[2]](https://ieeexplore.ieee.org/abstract/document/10232912)    | $$t=\lambda=\lambda_{+}^{\mathbb{Z}}+\lambda_{+}^{\mathbb{Q}\backslash\mathbb{Z}}-\lambda_{-}^{\mathbb{Z}}$$|
| `nonneg_bits_length`  |  The bit length assigned to the non-negative integer in the connection variable:    | **Symbol**: $$\lambda^{\mathbb{Z}}_{+}$$;<br>**Value**: Must be a positive integer number|
| `decimal_bits_length` |   The bit length assigned to the positive decimals in the connection variable   |**Symbol**: $$\lambda^{\mathbb{Q} \backslash \mathbb{Z}}_{+}$$;<br>**Value**: Must be a positive integer number|
| `negative_bits_length`|  The bit length assigned to the negative integer in the connection variable   | **Symbol**: $$\lambda^{\mathbb{Z}}_{-}$$;<br>**Value**: Must be a positive integer number |
| `submethod` |  Methods for solving subproblems    | keywords:<br>[`normal`](https://ieeexplore.ieee.org/abstract/document/10232912),<br>[`l_shape`](https://epubs.siam.org/doi/abs/10.1137/0117061)  |
|`debug_mode`|A flag to control whether LP files of both MAP and SUB will be printed or not| keywords:<br>`true`(print), <br>`false`(blank)|
|`Hybrid_mode`|A flag that controls whether to use a quantum computer to solve the MAP|keywords:<br>`true`(Q+C),<br>`false`(C+C)|
|`dwave`| parameters for D-Wave quantum machine||
|`mode`| The way to encode the MAP into the quantum solver| keywords:<br> *[`cqm`](https://docs.ocean.dwavesys.com/projects/system/en/stable/reference/samplers.html#dwave.system.samplers.LeapHybridCQMSampler)(LeapHybridCQMSampler),<br>[`bqm_hybrid`](https://docs.ocean.dwavesys.com/projects/system/en/stable/reference/samplers.html#dwave.system.samplers.LeapHybridSampler)(LeapHybridSampler),<br> [`bqm_quantum`](https://docs.ocean.dwavesys.com/projects/system/en/stable/reference/generated/dwave.system.samplers.DWaveSampler.sample.html#dwave-system-samplers-dwavesampler-sample)(DWaveSampler)<br> *preferred method|
|`DWave_token`|The special token for running the D-Wave| start with "DEV-xxxxxxxxxxxx"|
|`Mcut_flag`|A flag to control whether the muti-cuts strategy will be applied in the algorithm| keywords:<br>`true`(Yes), <br>`false`(No, only 1 cut per iteration)|
|`Cutnums`|The number of cuts per iteration when `Mcut_flag` is true (must greater than 0)||
|`num_of_read`|Indicates the number of states (output solutions) to read from the quantum solver. [(Ref)](https://docs.dwavesys.com/docs/latest/c_solver_parameters.html#num-reads)| Must be a positive integer in the range given by the [num_reads_range](https://docs.dwavesys.com/docs/latest/c_solver_properties.html#property-read-range) solver property|
|`threshold`|The stopping criterion of the algorithm||
|`type`|The type the stopping criterion|keywords:<br>`absolute`($$\epsilon\leq \mid\overline{\lambda} - \underline{\lambda}\mid$$), <br>`relative`($$\epsilon\leq \frac{\mid\overline{\lambda} - \underline{\lambda}\mid}{\mid\overline{\lambda}\mid}$$)|
|`gap`|The gap of the stopping criterion |Must be a positive number (e.g. 0.05)|
|`max_steps`|It will let the algorithm ends the loop after `max_steps` iteration|Must be a positive integer number|

## Contact
Zhongqi Zhao [zzhao27@uh.edu](mailto:zzhao27@uh.edu)

Mingze Li [mli44@central.uh.edu](mailto:mli44@central.uh.edu)

## Contributors
Zhongqi Zhao, Mingze Li, Lei Fan, Zhu Han.

## Citation

If you use HQCMCBD in your work, please cite our paper (will be there very soon)

<!---
```
@misc{zzhao2024hqcmbd,
  author    = {Kushnir, Sam and Leng, Jiaqi and Peng, Yuxiang and Fan, Lei and Wu, Xiaodi},
  publisher = {{INFORMS Journal on Computing}},
  title     = {{QHDOPT}: A Software for Nonlinear Optimization with {Q}uantum {H}amiltonian {D}escent},
  year      = {2024},
  doi       = {10.1287/ijoc.2024.0587.cd},
  url       = {https://github.com/INFORMSJoC/2024.0587},
  note      = {Available for download at https://github.com/INFORMSJoC/2024.0587},
}
```
>
