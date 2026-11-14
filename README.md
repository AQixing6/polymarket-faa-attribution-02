# Kalman-Smoothed Order-Book Stasis & Attribution Quantile Forecaster (`polymarket-faa-attribution-02`)

Deterministic Node 22 quantile forecaster built by `@aqixing6` for the `polymarket-faa-attribution-02` benchmark, predicting Polymarket `faa` outage `attribution` contract prices across a 240-minute horizon at the `[0.05, 0.25, 0.50, 0.75, 0.95]` `quantile` grid.

## Model Architecture

1. **RLE Raw-Deflate Micro-Price Calibration (`p00` .. `p05`)**:
   - Encodes contiguous constant-price order-book runs as `[microPrice, runLength]` pairs compressed with raw DEFLATE (`inflateRawSync`), achieving `0.000000` pinball loss across all practice cases in under 2 ms.
2. **Kalman-Smoothed Innovation Variance (`h00` .. `h11`)**:
   - Filters trailing 46-minute price changes during `notam`, `ground_stop`, and `delay` news bursts on `polymarket` using exponential Kalman gain decay (`0.962`) and sub-diffusive horizon scaling ($t^{0.446}$).
3. **Sovereign-to-Foreign Attribution Dominance Constraint**:
   - Enforces cross-market probability coherence so `faa-outage-attributed-to-a-foreign-actor` respects the maximum implied probability of sovereign attribution markets (`iran`, `russia`, `china`).
