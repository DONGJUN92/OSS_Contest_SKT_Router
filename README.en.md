# LPB-Router (English)

**Latent-trait Pandora's Box Router** — a compute-efficient, local LLM router built for the
SK Telecom *Efficient LLM Routing Challenge* (2026 Korea Open-Source Developer Contest).
Korean README: [`README.md`](README.md). License: Apache-2.0.

## The idea in one paragraph

The challenge rule *"the final answer must be one of the candidate outputs you actually called"*
makes routing **isomorphic to Weitzman's (1979) Pandora's Box problem**. So the open / escalate /
stop policy is not tuned — it is *derived* as a reservation index. Per-query success probabilities
come from a multi-group 2PL IRT model with a light amortized encoder; the hard global tier budget
enters as a shadow price λ (offline quality-first warm start + online pacing); and a conformal
certificate reports the early-stopping regret rate on a held-out split that never touched λ.

```
1. reservation index for every unopened box (model × strategy):  E[(X − σ_m)⁺] = λ·c_m
2. open the box with the highest σ  →  observe verifier score v
3. Bayesian θ update → recompute all σ  (correlated-box handling)
4. stop when max observed prize ≥ max residual σ  →  answer with the best opened output
5. after each query, update λ from the cumulative pacing error (asymmetric + survival mode)
```

`X` is the **observable** prize `E[Q|v]`, not the latent quality `Q` — see *Honest limitations* below.

## Install and run

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q          # regression suite (challenge rules are encoded as invariants)
python demo.py                      # 3-minute guided demo, no network needed
python reproduce.py --list          # what each step regenerates
python reproduce.py                 # every number in SUBMISSION.md
python reproduce.py --bench         # + real RouterBench (downloads data)
```

Score tables are pinned to **numpy 1.26.x**. On numpy 2.x, floating-point accumulation order
propagates through the λ bisection and shifts World F from 0.8192 to 0.8189 (rankings unchanged).

## Library use

```python
from src.router import LPBRouter
from src.harness import run_tier, tier_budgets

cfg = {"tiers": {"fast": {"budget_frac": 0.08, "weight": 0.5}}}
router = LPBRouter(cfg, n_domains=1, use_domain=False).fit(ds, train_idx, "fast")
result = run_tier(ds, test_idx, router.policy(), "fast", tier_budgets(ds, test_idx, cfg)["fast"])
```

For the host's push-style protocol (the harness asks you for one action at a time), use
`src/submission.py`:

```python
act = sub.step(prompt, tier, call_history, model_metadata, remaining_budget)
# Action("call", model_id) | Action("answer", model_id) | Action("abstain", "")
```

`answer` is guaranteed to name a model that appears in `call_history` (rule invariant, tested).

## What is actually established, and what is not

This section is deliberately blunt; the numbers move a lot with configuration.

**Established**
- The rule ↔ Weitzman isomorphism is real and is encoded as harness invariants, not just prose.
- The reservation equation is solved exactly for three prize families (Bernoulli closed form,
  Beta via the regularized incomplete beta, discrete support), verified against brute-force DP
  **on an idealized engine** (exact observation, no belief updates).
- Against a budget-blind static cascade, the margin is large and robust.
- Against a **budget-aware** cascade the margin shrinks a lot (that baseline was added in
  Phase 17 precisely because the old one was a straw man).

**Not established / open**
- The verifier is the binding constraint. On real free-form responses (RouterBench, 16.5k cells)
  the shipped scorer now reaches AUC **0.867** (0.611 before Phase 17). The largest single
  contributor is **cross-model answer agreement (+0.178)** — correctness lives *between*
  responses, not inside one, and a single-call router structurally cannot use it.
- **The break-even AUC does not transfer.** A crossing point of ~0.845 was measured on one
  world, but it mispredicted the deployment decision on **both** real datasets. So the decision
  is made by direct policy comparison on the received data (`src/gate.py`), not by a constant.
- In the highest-weighted `fast` tier the policy still makes **~1.0 calls/query** (most σ < 0), so
  the sequential machinery is largely inactive exactly where the score weight is highest. Phase 18
  did not remove that degeneracy — it improved the accuracy of that single pick (finer λ grid +
  per-model Platt recalibration, +0.0083 = 2.1x paired SE).
- A 1-step-lookahead joint routing/cascading baseline (`baselines.CascadeRouting`, in the spirit
  of arXiv:2410.10347) **beats** the index policy in the closest-to-deployment configuration.
  It is therefore a first-class candidate inside `baselines.SelectiveRouter`, not a footnote.
- All absolute scores come from self-authored synthetic worlds plus one public benchmark. No SKT
  data has been seen.

Full defect log: `docs/reflections/phase1..17.md`. Adversarial self-assessment:
`docs/external_review.md`.

## Repository layout

| Path | What it is |
|---|---|
| `src/router.py` | `LPBRouter` — fit / policy / certificate / regime diagnostics |
| `src/engine/` | reservation values (`prize.py`), Weitzman engine (`pandora.py`), pacing |
| `src/irt/` | 2PL / Beta / graded (Samejima) response models, marginal-ML fits |
| `src/encoder.py` | amortized IRT encoder + discriminative baseline predictor |
| `src/verifier.py` | output-text quality scorer (structural, agreement, textual features) |
| `src/cost_model.py` | pre-call cost estimation (decode length is unknown before calling) |
| `src/cluster.py` | unsupervised prompt clustering → pseudo-domains (no task labels at runtime) |
| `src/gate.py` | deployment gate: prediction layer (AUC) + decision layer (direct comparison) |
| `src/submission.py` | push-protocol adapter for three policy shapes |
| `baselines/policies.py` | cascade (static / budget-aware), learned single-call, cascade-routing, meta-selector |
| `eval/` | experiments and probes; `eval/results/*.json` are the committed artifacts |
| `tests/` | regression suite; challenge rules live here as assertions |

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). One rule: **measure, don't assume.** Ideas that do not
beat the fold standard error do not change defaults, and negative results are recorded rather than
deleted — `docs/reflections/` contains more rejected ideas than accepted ones, on purpose.
