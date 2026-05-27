# social-media-screening

An experiment that screens line-aligned AAVE/SAE samples for sexually
explicit content, producing a paired CSV intended as input to an
external content-moderation tool whose racial bias you want to audit.

## What it does

`local/screen_pairs.py`:

1. Reads two line-aligned text files (`local/aave_samples.txt` and
   `local/sae_samples.txt`) of AAVE tweets and their SAE rewrites.
2. Runs a local multi-label moderation model
   ([`unitary/unbiased-toxic-roberta`](https://huggingface.co/unitary/unbiased-toxic-roberta),
   the Detoxify "unbiased" checkpoint) over the AAVE lines.
3. Filters on one Detoxify label at a time (default `sexual_explicit`;
   pass `--label obscene` for profanity, or any other Detoxify label).
   Filtering on a single label — rather than a combined NSFW score —
   keeps profanity, slurs, and AAVE features from flagging a row on
   the wrong axis.
4. Writes flagged rows to `local/results_<label>.csv` alongside their
   SAE counterparts.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the Detoxify model (~500 MB) from HuggingFace
into the local cache. Inference uses CUDA, Apple Silicon (MPS), or CPU,
auto-selected in that order.

## Run

```bash
python local/screen_pairs.py                    # → local/results_sexual_explicit.csv
python local/screen_pairs.py --label obscene    # → local/results_obscene.csv (profanity)
```

`--threshold` (default `0.5`) sets the minimum label probability for a
row to be kept. `--label` accepts any Detoxify label: `toxicity`,
`severe_toxicity`, `obscene`, `identity_attack`, `insult`, `threat`,
`sexual_explicit`.

## Output

`local/results_<label>.csv` columns:

| column | description |
|---|---|
| `line_index` | 0-based line index in the input files |
| `<label>_score` | probability for the chosen Detoxify label (0–1) |
| `aave_text` | Original AAVE line |
| `sae_text` | SAE rewrite |

Feed `aave_text` and `sae_text` into the external moderation tool you're
auditing and compare flagging rates.

## Notes

- The moderation model inherits biases from its training data (Jigsaw
  "Unintended Bias in Toxicity Classification"). Spot-check flagged
  rows before drawing conclusions about the audit dataset.
- For background on dialect bias in toxicity / hate-speech classifiers,
  see Sap et al. 2019 (*The Risk of Racial Bias in Hate Speech
  Detection*) and Hofmann et al. 2024 (*Dialect prejudice predicts AI
  decisions about people's character, employability, and criminality*).
