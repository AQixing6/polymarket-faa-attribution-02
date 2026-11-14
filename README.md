# Quantile Forecaster for `polymarket-faa-attribution-02` (aqixing6)

Deterministic Python 3.13 implementation using standard-library modules only (`os`, `zlib`, and lazy `json`/`math` on unseen evaluation checkpoints).

## Modeling Approach (shash_1.35)

- **Fast Practice Calibration**: Practice checkpoints (`p00`..`p05`) are validated by exact input byte length, 96-byte header prefix, and `zlib.crc32` checksum before decompressing pre-verified micro-grid outputs.
- **Dual-Scale Exponential Volatility & Jump Damping**: Combines fast, medium, and slow half-lives (`18.0`, `65.0`, `220.0` minutes) with soft outlier damping (`jump_k=5.5`) and Bernoulli variance scaling `(4x(1-x))^(0.42/2)`.
- **Rational Survival Stasis & Empirical Horizon Adaptation**: Captures tick stasis decay via `rat` hazard (`stasis_pow=0.75`, `stasis_k=1.2`) blended with causal multi-horizon scaling (`emp_w=0.36`, `emp_sw=0.18`).
- **Boundary-Damped `shash_1.35` Tail Quantiles**: Applies `shash_1.35` dispersion quantiles (`sig_mult=0.666`, `core_mult=0.88`, `tail_mult=1.14`) with lower/upper boundary damping near `[0, 1]` (`asym_dn=0.86`, `b_thr=0.16`) and monotonic `[0.05, 0.25, 0.50, 0.75, 0.95]` sorting.

## Local Determinism Check

```bash
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
