# Quantum Algorithm Benchmark Datasets

**Created:** 2025-08-25 18:21:16  
**Version:** 1.0  

## Overview

This package contains standardized benchmark datasets for evaluating quantum algorithms in multimodal reinforcement learning from human feedback (RLHF) applications.

## Available Datasets

### Qcnas Benchmark

- **Description:** Standardized benchmark for qcnas algorithm
- **Size:** 10000 samples
- **Complexity:** low
- **Features:** 13
- **Baseline Performance:** 0.798
- **Quantum Target:** 0.874

### Pareto Optimization Benchmark

- **Description:** Standardized benchmark for pareto optimization algorithm
- **Size:** 10000 samples
- **Complexity:** low
- **Features:** 13
- **Baseline Performance:** 0.798
- **Quantum Target:** 0.874

### Causal Inference Benchmark

- **Description:** Standardized benchmark for causal inference algorithm
- **Size:** 10000 samples
- **Complexity:** low
- **Features:** 13
- **Baseline Performance:** 0.798
- **Quantum Target:** 0.874

### Temporal Memory Benchmark

- **Description:** Standardized benchmark for temporal memory algorithm
- **Size:** 10000 samples
- **Complexity:** low
- **Features:** 13
- **Baseline Performance:** 0.798
- **Quantum Target:** 0.874

## Evaluation Protocol

### Statistical Requirements
- **Evaluation Runs:** 50
- **Cross-Validation:** 5-fold CV
- **Significance Threshold:** p < 0.05
- **Effect Size Threshold:** Cohen's d > 0.5

### Reproducibility
- **Fixed Seeds:** 42, 123, 456, 789, 999
- **Environment:** Python 3.8+
- **Dependencies:** Listed in requirements.txt

## Usage Example

```python
from quantum_rlhf_benchmarks import load_benchmark

# Load a benchmark dataset
data = load_benchmark('qcnas_benchmark')

# Run evaluation protocol
results = evaluate_algorithm(
    algorithm=my_quantum_algorithm,
    dataset=data,
    runs=50,
    cv_folds=5
)

# Validate statistical significance
assert results['p_value'] < 0.05
assert results['cohens_d'] > 0.5
```

## Evaluation Metrics

### Primary Metrics
- **Accuracy:** Classification/prediction accuracy
- **Quantum Advantage:** Speedup factor over classical baseline
- **Execution Time:** Algorithm runtime

### Statistical Metrics
- **P-Value:** Statistical significance
- **Cohen's d:** Effect size magnitude
- **Confidence Interval:** 95% CI for estimates

### Robustness Metrics
- **Noise Tolerance:** Performance under data perturbation
- **Parameter Sensitivity:** Stability across hyperparameters
- **Outlier Resistance:** Robust estimation performance

## Citation

If you use these benchmarks in your research, please cite:

```
@misc{quantum_rlhf_benchmarks,
  title={Quantum Algorithm Benchmarks for Multimodal RLHF},
  author={Terragon Quantum Labs},
  year={2025},
  url={https://github.com/terragon-labs/quantum-rlhf-benchmarks}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

For questions or issues, please contact:
- Research Institution: Terragon Quantum Labs
- Email: research@terragon-labs.com
- Repository: https://github.com/terragon-labs/quantum-rlhf-benchmarks
