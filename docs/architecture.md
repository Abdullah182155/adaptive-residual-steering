# ARS architecture

The maintained study adds small residual-stream modules to a frozen `microsoft/phi-2` model. It evaluates a fixed-route adapter and a separate bank of route-specific adapters. A prompt-conditioned router can select an action from the bank at inference time.

## System path

```text
Question and optional demonstrations
        |
        v
Frozen Phi-2 backbone
        |
        +-- fixed route: layers 12, 15, 18, 21
        |       `-- one fixed-route low-rank residual adapter
        |
        `-- adaptive route:
                prompt-only route-value router
                    `-- selected bank expert and gain
        |
        v
Generated answer and recorded metrics
```

The fixed route and bank route are independent experiment arms. A route is a set of backbone layer indices, not the number of internal layers inside an adapter.

## Fixed-route adapter

The Stage A checkpoint uses architecture ID `ars.fixed_route_low_rank_residual`, architecture version `1.0.0`, route `[12, 15, 18, 21]`, rank 16, and `rho_max=0.10`. It uses fresh per-layer p99 norm calibration and has no learned magnitude head. The backbone remains frozen. Each adapter projects a selected layer's hidden state into a low-rank space, applies a GELU nonlinearity, projects back to the residual width, applies the configured gain and calibrated cap, then adds the update to the residual stream.

The run-specific gain is calibrated separately in Stage B. At evaluation, the fixed-steering arm uses the resulting selected gain consistently for its scored questions.

## Independent route bank

The bank is built from the candidate-layer list recorded in the Stage A manifest. The current Stage A list contains 14 layer locations. The catalog can contain `k=0` (no intervention) and route experts with one to four active layers. Candidate multi-layer routes use the configured spacing rules and bank caps.

Each route has its own adapter parameters and checkpoint entry. Singleton experts are trained first. Larger route experts are initialized from measured trained subsets and then fine-tuned as route-specific copies. The fixed-route adapter checkpoint does not initialize the bank. Each expert uses the same frozen Phi-2 backbone, matching norm references, and its own layer subset.

The bank objective combines token-mean likelihood on verified positive completions with a weighted softplus pairwise margin on same-prompt correct and well-formatted wrong completions when a matched pair exists. The current CLI default for the preference term is `0.25`. Held-out route screening uses pairwise preference evidence where available and positive-completion likelihood as a fallback. Route caps constrain the search workload; they do not guarantee that every retained route will be useful or selected.

## Static action and router

The static-bank arm chooses one route and gain on held-out data, then reuses that action for every benchmark question. This separates bank quality from per-question routing.

The router is a `RouteValueRouter`. It scores a finite catalog of route-and-gain actions, including the no-steering action. Its inputs contain a pooled representation of valid prompt-token hidden states, normalized shot count, and normalized question-token count. It does not receive generated completion tokens or the gold answer as inference features. The router chooses the lowest predicted cost-adjusted action score. The action score incorporates route cost `0.01 × k` during teacher construction and evaluation.

During training, the router learns from per-question measured action scores. The training loss combines smooth-L1 score regression with a KL preference loss over the teacher's action ranking. Validation checks likelihood and paired answer behavior. When task-validation examples are available, checkpoint selection uses held-out task accuracy against the static action and the unsteered baseline.

## Compatibility boundary

Stage A's handoff manifest pins the adapter architecture, route, candidate layers, calibration files, rollout hashes, and code commit. Stage B checks those identities before training. The final evaluation bundle contains the fixed adapter, independent bank, router, both calibration files, and the validation-selected static action. A checkpoint from an older prototype is not interchangeable merely because it is also a PyTorch file.

The repository retains older gating, policy-gradient, and prompt-conditioned experiments for historical or compatibility use. They are not the maintained final-study path described above.

## Measured outcome of the current study

On the available matched GSM8K, GSM-Plus, and ARC-Challenge evaluation, the fixed adapter improves 0-shot accuracy over baseline on all three datasets. The current static bank and router also exceed baseline in 0-shot point accuracy, but they do not exceed the fixed adapter. At 2-shot, the saved paired analysis does not establish a reliable improvement. The router does select multiple actions, but its use is concentrated and no four-layer action appears in the six tested conditions. See [final results](final_results.md) for exact deltas and [limitations](limitations.md) for interpretation boundaries.

## Diagrams

Editable Draw.io sources and matching PNGs are in [`diagrams/`](../diagrams/README.md). The research deck pairs each architecture diagram with neighboring slides that explain the model, training objective, validation choice, or evaluation role it represents.
