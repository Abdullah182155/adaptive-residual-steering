# Final-study diagrams

The seven figures below document the maintained final-study design. Every PNG has a matching editable Draw.io source with the same filename stem. The deck uses these figures alongside slides that explain the training choices and their interpretation.

| Diagram | PNG | Draw.io source | What it explains |
|---|---|---|---|
| Full system | `diagram_ars_full_system.png` | `diagram_ars_full_system.drawio` | Verified GSM8K data, fixed adapter, bank, prompt router, and four-arm evaluation |
| Rollout collection | `diagram_rollout_collection.png` | `diagram_rollout_collection.drawio` | Initial attempts, strict-positive verification, targeted retries, deduplication, and training rows |
| Fixed adapter | `diagram_steernet.png` | `diagram_steernet.drawio` | Low-rank residual update, fixed route, calibrated gain, p99 norm cap, and residual addition |
| Route bank | `diagram_route_bank.png` | `diagram_route_bank.drawio` | Candidate layers, spacing, singleton-to-quad curriculum, independent experts, objective, and selection |
| Steering policies | `diagram_static_vs_dynamic.png` | `diagram_static_vs_dynamic.drawio` | Fixed steering, one static bank action, and per-question router decisions |
| Router | `diagram_router.png` | `diagram_router.drawio` | Offline teacher scoring and training separated from prompt-only inference |
| Paired evaluation | `diagram_paired_evaluation.png` | `diagram_paired_evaluation.drawio` | Shared test inputs, four arms, paired results, quality checks, and runtime metrics |

The source of truth for both formats is [`scripts/generate_study_diagrams.py`](../scripts/generate_study_diagrams.py). Regenerate the complete set from the repository root:

```powershell
python scripts/generate_study_diagrams.py
```

The figures use a black background with orange and white elements to match the research presentation. The architecture diagrams explain the protocol; benchmark results belong only in separately labeled evaluation slides.
