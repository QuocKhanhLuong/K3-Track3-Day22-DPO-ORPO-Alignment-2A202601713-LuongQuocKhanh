# Preference Alignment Experiment Report

## 1. Dataset Analysis & Cleaning

### Data Loading Summary
- **Total examples loaded**: `24`
- **Validation issues found**: Line 1 contained unescaped quotes around `"self-attention"`, so the JSONL parser failed before validation.
- **Cleaning steps taken**: Escaped the inner quotes on line 1; changed `load_jsonl()` to report file/line numbers for JSON and schema failures; reject duplicate prompts after case/whitespace normalization; and strengthened chosen/rejected validation.

### Split Strategy
- **Train/Val Ratio**: Target `80/20`; for the 24 unique sample prompts this produces 19 train prompts and 5 validation prompts.
- **Leakage Prevention**: `split_by_prompt()` groups records by normalized prompt, shuffles prompt groups deterministically with `random.Random(seed)`, then assigns whole groups to one side only. Tests assert that train/validation prompt sets are disjoint and that all examples are preserved.

## 2. Implementation: DPO (plus ORPO bonus)

### Objective Selection
- **Why this method?**: DPO is the primary method because the provided config uses `training.method: dpo` and the lab focuses on comparing the policy preference margin against a frozen reference model. ORPO was also implemented as a bonus to verify the reference-free odds-ratio objective.
- **Key Hyperparameters**:
  - `beta`: `0.1`
  - `lambda_orpo`: `0.1`

### Numerical Stability
- **Challenges**: Directly computing `log(sigmoid(x))` can overflow for very negative margins. ORPO can also hit `log(0)` when a log-probability is exactly `0.0` (`p=1`).
- **Solutions**: Implemented `log(sigmoid(x))` as `-np.logaddexp(0, -x)`. ORPO clips log-probabilities to `[-30.0, -1e-7]` before computing log-odds with `log1p`. Unit tests cover extreme DPO inputs and ORPO at `logp=0.0`.

## 3. Evaluation Results

### Metrics

`outputs/metrics.json` produced by the deterministic CPU scorer:

```json
{
  "pairwise_accuracy": 0.9166666666666666
}
```

| Metric | Value |
|---|---:|
| Pairwise Accuracy | `91.67%` |
| DPO closed-form unit-test loss | `0.663597` |
| ORPO closed-form unit-test loss | `1.017086` |
| Final trained loss | `N/A` — the required lab path does not train a model |

Tie handling is explicit: a tie contributes `0.5` win. Score-list length mismatches raise `ValueError` instead of being silently truncated.

### Qualitative Review
- **Prompt**: `Describe the bias-variance tradeoff in machine learning.`
- **Chosen Response**: `The bias-variance tradeoff is a fundamental concept where reducing bias often increases variance, and vice versa. High bias models are too simple and underfit, while high variance models are too complex and overfit.`
- **Rejected Response**: `The bias-variance tradeoff is a marketing term used to describe the difficulty of choosing the right hyperparameters.`
- **Heuristic Preference**: **Incorrect** — chosen score about `0.517`, rejected score about `0.553`.

The scorer is deterministic and content-aware: it combines prompt keyword coverage, unique informative-token count, a concision prior, and a repetition penalty. Re-running evaluation gives the same metric because no randomness is used.

## 4. Discussion & Failure Modes

- **What went well?**: The full required path now has line-aware data validation, leakage-safe splitting, numerically stable DPO/ORPO losses, explicit evaluation semantics, and a reproducible scorer instead of hard-coded `1.0/0.0` scores.
- **Observed Bias / Failure mode**: The heuristic scorer is still lexical and length-sensitive. It mis-scores 2 of 24 pairs. In the bias-variance example above, the concise rejected answer repeats prompt terms efficiently and beats the longer correct explanation. A similar failure occurs on the dropout example. Therefore `91.67%` measures this scorer's agreement with the preference labels, not model intelligence or factual correctness.
- **Safety / Regression prompts**: No generative model is trained or invoked on the required CPU path, so the four regression prompts in `docs/regression_prompts.md` cannot be truthfully executed as generation tests. Qualitative expectations are: high-risk medical advice should defer to a professional; strict summaries should obey the word limit; uncertain questions should admit uncertainty; and troubleshooting with missing context should ask for clarification. This is recorded as a limitation rather than fabricated model evidence.
