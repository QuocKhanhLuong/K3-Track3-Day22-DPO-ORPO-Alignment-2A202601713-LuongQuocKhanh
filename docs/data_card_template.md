# Data Card

- **Dataset name:** Day22 sample preference pairs
- **Source:** Course-provided `data/sample_preferences.jsonl` bundled with this repository; 24 English machine-learning education prompts.
- **License/permission:** Provided for use in this lab/repository. No separate dataset license is stated in the starter materials, so no broader redistribution permission is assumed.
- **Schema:** JSONL records with `prompt: str`, `chosen: str`, `rejected: str`, and optional `metadata: dict`; each row is validated by `PreferenceExample`.
- **Labeling rubric:** Pairwise preference under the `accuracy` rubric: `chosen` should be more technically correct/helpful than `rejected`; the label is relative rather than an absolute quality score.
- **Known biases:** Narrow English ML-education domain; rejected answers are often shorter and contain plausible terminology, creating length/lexical-overlap bias for simple scorers. The dataset is small and not representative of general alignment behavior.
- **Safety/PII checks:** No explicit personal data is present in the bundled sample on manual review. Loader validation rejects malformed/schema-invalid rows and duplicate prompts. PII detection is not part of the required path and should be added before using external/private data.
- **Train/validation/test split method:** Deterministic prompt-group split with seed `42`; target validation ratio `0.2`. All records sharing the same normalized prompt are assigned to exactly one split, preventing prompt leakage. No held-out test split is supplied by the lab.
