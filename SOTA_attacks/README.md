# SOTA (State-of-the-Art) Attacks

This directory contains wrappers for published reconstruction attack implementations, integrating them into our standard experiment framework while maintaining links to their original repositories.

## Available Attacks

### Linear Reconstruction Attack

**Paper**: ["A Linear Reconstruction Approach for Attribute Inference Attacks against Synthetic Data"](https://arxiv.org/abs/2301.10053)
**Authors**: M.S.M.S. Annamalai, A. Gadotti, L. Rocher
**Conference**: USENIX Security 2024
**Source Repository**: https://github.com/steveng9/recon-synth (forked from https://github.com/Filienko/recon-synth)

#### How It Works

The attack uses k-way marginal queries combined with Linear Programming (LP) to reconstruct secret binary attributes from synthetic data. It queries the synthetic data about combinations of attributes and uses the results to infer the training data's sensitive features.

#### Requirements

- **Gurobi**: Academic license required (free for academics)
- **Feature Type**: Binary categorical features only (2 unique values)
- **Dataset Size**: Works best with n < 5000 for 3-way queries (memory constraints)
- **Single Feature**: Reconstructs one feature at a time

#### Usage in Experiments

In your config YAML:

```yaml
attack_method: "LinearReconstruction"
data_type: "categorical"

attack_params:
  LinearReconstruction:
    k: 3  # k-way queries (2 or 3 recommended)
    n_procs: 4  # Number of processors for Gurobi

  # Chaining and ensembling NOT recommended
  # (attack is designed for single features)
  ensembling:
    enabled: false
  chaining:
    enabled: false
```

#### Example Config

See `configs/test_linear_attack.yaml` for a complete example.

#### Important Constraints

1. **Single Binary Feature**: The attack only works when `hidden_features` has exactly 1 element that is binary
2. **Requires ALL Features as QI**: For best results, QI should include ALL non-secret features. The attack generates queries over all attributes, and limiting to a small QI subset significantly reduces accuracy
   - ✅ Good: QI = [all 24 features except secret] → 78% accuracy
   - ❌ Poor: QI = [3 features] → 67% accuracy (below baseline)
3. **Memory Usage**: Query matrix size grows as O(C(d,k-1) * values_per_attr * n), where d=num_attributes
4. **Computational Cost**: LP solving with Gurobi can be slow for large query matrices
5. **No Chaining**: Don't use with chaining enhancement (designed for single features)
6. **No Ensembling**: Not compatible with ensembling (different attack paradigm)

#### Testing

Run the integration test:
```bash
cd /home/golobs/Reconstruction
python maintenance_scripts/test_linear_integration.py
```

#### Implementation Details

The wrapper (`linear_reconstruction.py`) handles:
- Converting DataFrames to the format expected by recon-synth
- Binary value mapping (e.g., {1,2} → {0,1})
- Query generation and scaling
- LP solving via Gurobi
- Converting results back to standard DataFrame format
- Validation and error checking

## Adding New SOTA Attacks

To add a new published attack:

1. Create a new wrapper module in this directory
2. Implement the standard attack interface:
   ```python
   def attack_name(cfg, synth, targets, qi, hidden_features):
       # ... implementation ...
       return reconstructed, None, None
   ```
3. Register in `attacks/__init__.py` ATTACK_REGISTRY
4. Add documentation here
5. Create a test config in `configs/`
6. Add test in `maintenance_scripts/`

## Citation

If you use the Linear Reconstruction attack in your research, please cite:

```bibtex
@inproceedings{annamalai2024linear,
  title={A Linear Reconstruction Approach for Attribute Inference Attacks against Synthetic Data},
  author={Annamalai, M.S.M.S. and Gadotti, A. and Rocher, L.},
  booktitle={33rd USENIX Security Symposium},
  year={2024}
}
```
