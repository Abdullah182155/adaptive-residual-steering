# Adaptive Residual Steering

ARS is a compact Phase 1 research implementation for steering `microsoft/phi-2` on GSM8K. It uses one certified, fixed route: layers **13 → 20 → 22** with SteerNet V5 residual adapters.

```text
prompt ──► Phi-2 ──► [L13 V5] ──► [L20 V5] ──► [L22 V5] ──► answer
                    fixed route; router and learned gate are intentionally absent
```

V5 uses a rank-64 bottleneck, causal depthwise convolution with warm-cache hand-off, Dirac-centred current-token initialization, and calibrated per-layer norm caps. Few-shot demonstrations are shielded while the question remains steered during prefill. Deployment energy is 1.00/0.50/0.35 for 0/2/4 shots.

```bash
python scripts/calibrate_norms.py --output norms_calibration.json
python scripts/run_ars_pipeline.py --stage all
pytest tests/ -q -m "not slow"
```

`experiments/history.json` is the run ledger. Historical presentations and rendered diagrams remain local but are intentionally not tracked.
