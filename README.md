# HQCMCBD

[![Project page](https://img.shields.io/badge/project%20page-online-c8102e)](https://djzts.github.io/HQCMCBD-API/)
[![IEEE QCNC 2025](https://img.shields.io/badge/IEEE-QCNC%202025-00629b)](#citation)
[![License: MIT](https://img.shields.io/badge/license-MIT-2f7d32)](LICENSE)
[![Video tutorial](https://img.shields.io/badge/video-tutorial-red)](https://www.youtube.com/watch?v=VcDaOj8W4MI)

Hybrid Quantum-Classical Multi-cuts Benders' Decomposition for mixed-integer linear programming.

HQCMCBD is a Python implementation of the Hybrid Quantum-Classical Multi-cuts Benders' Decomposition algorithm. It solves mixed-integer linear programming (MILP) problems by decomposing them into a binary master problem and continuous subproblems. The master problem can be solved classically with Gurobi or encoded for quantum and quantum-inspired solvers such as D-Wave CQM/BQM, OpenJij, and simulated bifurcation.

**Project page: <https://djzts.github.io/HQCMCBD-API/>**. It explains the algorithm (master problem, solver backends, normal and L-shaped cuts, multi-cuts), walks through a quick start, shows results, and includes an interactive `config.json` builder.

Tutorial: [HQCMCBD overview video](https://www.youtube.com/watch?v=VcDaOj8W4MI)

## Features

- Classical Benders decomposition workflow with Gurobi.
- Hybrid quantum-classical master problem solving through D-Wave CQM/BQM samplers.
- Local quantum-inspired solving through OpenJij and simulated bifurcation.
- Single-cut and multi-cut modes.
- Configurable stopping criteria, lambda encoding precision, solver backend, and debug output.
- Notebook examples for quick experimentation.

## Requirements

- Python 3.8 or later
- Gurobi 9.1.2 or later with a valid license
- `gurobipy`
- `numpy`
- `dimod`
- `dwave-system`
- Optional: `openjij` for OpenJij mode
- Optional: `torch` and `simulated-bifurcation` for simulated bifurcation mode

The repository currently does not include a `requirements.txt`, so install the solver packages required by the modes you plan to use.

## Project Layout

| Path | Purpose |
| --- | --- |
| `HQCMCBD.py` | Main implementation. |
| `HQCMCBD_notebook.ipynb` | Notebook loader that delegates to `HQCMCBD.py`. |
| `Example.ipynb` | Basic example notebook. |
| `Example-3bin-run.ipynb` | Three-binary-variable example. |
| `Example-3bin-output.ipynb` | Example notebook with output. |
| `config.json` | User-editable configuration for `mode="manual"`. |
| `config_default.json` | Default configuration for `mode="default"`. |
| `data_output/` | Runtime solution and metadata output. |
| `LP_output/` | Debug LP exports from Gurobi models. |
| `LP_output_dwave/` | Quantum and quantum-inspired solver output files. |
| `docs/` | Source of the project page, published to GitHub Pages. |

## Quick Start

Import the solver in a Python script:

```python
from HQCMCBD import HQCMCBD_algorithm
```

Or load it in a notebook:

```python
%run HQCMCBD_notebook.ipynb
```

Create a Gurobi model, then pass it to HQCMCBD:

```python
import gurobipy as gp
from gurobipy import GRB
from HQCMCBD import HQCMCBD_algorithm

model = gp.Model("example_model")

# Build your MILP model here.
# Add binary variables, continuous non-negative variables, constraints,
# and the objective function.

model.optimize()

solver = HQCMCBD_algorithm(model, mode="manual")
solver.run()
```

Use `mode="manual"` to load `config.json`, or `mode="default"` to load `config_default.json`.

## Configuration

Before running the solver, review the selected configuration file.

```json
{
    "lambda_var": {
        "nonneg_bits_length": 15,
        "decimal_bits_length": 4,
        "negative_bits_length": 15
    },
    "submethod": "l_shape",
    "debug_mode": false,
    "Hybrid_mode": false,
    "dwave": {
        "mode": "cqm",
        "DWave_token": "YOUR_DWAVE_TOKEN",
        "Mcut_flag": false,
        "Cutnums": 1,
        "num_of_read": 100
    },
    "threshold": {
        "type": "absolute",
        "gap": 0.5
    },
    "max_steps": 20
}
```

| Key | Description | Valid values |
| --- | --- | --- |
| `lambda_var.nonneg_bits_length` | Bit length for the non-negative integer component of the connection variable. | Positive integer |
| `lambda_var.decimal_bits_length` | Bit length for the positive fractional component of the connection variable. | Positive integer |
| `lambda_var.negative_bits_length` | Bit length for the negative integer component of the connection variable. | Positive integer |
| `submethod` | Subproblem strategy. | `normal`, `l_shape` |
| `debug_mode` | Whether to export LP model files. | `true`, `false` |
| `Hybrid_mode` | Whether to solve the master problem with a quantum or quantum-inspired backend. | `true`, `false` |
| `dwave.mode` | Master-problem backend when `Hybrid_mode` is enabled. | `cqm`, `bqm_hybrid`, `bqm_quantum`, `bifurcation`, `openjij` |
| `dwave.DWave_token` | D-Wave API token. | Token string or `YOUR_DWAVE_TOKEN` |
| `dwave.Mcut_flag` | Whether to add multiple cuts per iteration. | `true`, `false` |
| `dwave.Cutnums` | Number of cuts when multi-cut mode is enabled. | Positive integer |
| `dwave.num_of_read` | Number of samples for BQM/OpenJij modes. | Positive integer |
| `threshold.type` | Stopping criterion type. | `absolute`, `relative` |
| `threshold.gap` | Stopping gap. | Positive number |
| `max_steps` | Maximum Benders iterations. | Positive integer |

For D-Wave runs, do not commit real tokens. Either put the token in a local, untracked config file or set:

```powershell
$env:DWAVE_API_TOKEN = "YOUR_TOKEN"
```

## Outputs

Runtime files are written to:

- `data_output/` for objective values, lambda bounds, solver metadata, and MAP solutions.
- `LP_output/` for Gurobi LP exports when `debug_mode` is enabled.
- `LP_output_dwave/` for quantum and quantum-inspired solver artifacts.

These files are useful for debugging and reproducing experiments, but they may be regenerated by future runs.

## Notes

- Continuous variables should be non-negative. The preprocessing step warns when a variable has a negative lower bound.
- Maximization models are converted internally to minimization and reported back with the original sign convention.
- `HQCMCBD_notebook.ipynb` intentionally loads `HQCMCBD.py` instead of duplicating the implementation.

## Research Using HQC-Bend

Peer-reviewed work that applies the hybrid quantum-classical Benders' decomposition method:

- M. Li, S. Wang, L. Fan, Z. Han, "[Quantum assisted combinatorial Benders' algorithm for the synergy of hydrogen and power distribution systems with mobile storage](https://ieeexplore.ieee.org/abstract/document/10841900/)," *IEEE Transactions on Power Systems*, vol. 40, no. 5, pp. 3967–3978, 2025.
- Y. Zhang, Y. Gong, L. Fan, Y. Wang, Z. Han, Y. Guo, "[Quantum-Assisted Online Task Offloading and Resource Allocation in MEC-Enabled Satellite-Aerial-Terrestrial Integrated Networks](https://ieeexplore.ieee.org/abstract/document/10804104/)," *IEEE Transactions on Mobile Computing*, 2024.
- Y. Zhang, Y. Gong, L. Fan, Y. Wang, Z. Han, Y. Guo, "[Quantum-assisted joint virtual network function deployment and maximum flow routing for space information networks](https://ieeexplore.ieee.org/abstract/document/10691643/)," *IEEE Transactions on Mobile Computing*, vol. 24, no. 2, pp. 830–844, 2025.
- X. Wei, L. Fan, Y. Guo, Y. Gong, Z. Han, Y. Wang, "[Hybrid quantum–classical Benders' decomposition for federated learning scheduling in distributed networks](https://ieeexplore.ieee.org/abstract/document/10636852/)," *IEEE Transactions on Network Science and Engineering*, vol. 11, no. 6, pp. 6038–6051, 2024.
- Y. Zhang, Y. Gong, L. Fan, Y. Wang, Z. Han, Y. Guo, "[Quantum-assisted joint caching and power allocation for integrated satellite-terrestrial networks](https://ieeexplore.ieee.org/abstract/document/10614917/)," *IEEE Transactions on Network Science and Engineering*, vol. 11, no. 6, pp. 5163–5174, 2024.
- X. Wei, L. Fan, Y. Guo, Y. Gong, Z. Han, Y. Wang, "[Quantum assisted scheduling algorithm for federated learning in distributed networks](https://ieeexplore.ieee.org/abstract/document/10230094/)," *2023 32nd International Conference on Computer Communications and Networks (ICCCN)*, IEEE, pp. 1–10, 2023.

## Contact

- Zhongqi Zhao: [zzhao27@uh.edu](mailto:zzhao27@uh.edu)
- Mingze Li: [mli44@central.uh.edu](mailto:mli44@central.uh.edu)

## Contributors

Zhongqi Zhao, Mingze Li, Lei Fan, Zhu Han.

## Citation

If you use HQCMCBD in your work, please cite the [QCNC 2025 paper](https://ieeexplore.ieee.org/abstract/document/11000177/):

```bibtex
@inproceedings{zhao2025hqc,
  title={HQC-Bend: A Python Package of Hybrid Quantum-Classical Multi-cuts Benders' Decomposition Algorithm},
  author={Zhao, Zhongqi and Li, Mingze and Fan, Lei and Han, Zhu},
  booktitle={2025 International Conference on Quantum Communications, Networking, and Computing (QCNC)},
  pages={591--597},
  year={2025},
  organization={IEEE},
  url={https://github.com/djzts/HQCMCBD-API},
  note={Available for download at https://github.com/djzts/HQCMCBD-API}
}
```

## License

This project is released under the license included in [LICENSE](LICENSE).
