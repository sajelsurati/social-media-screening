# social-media-screening

A three-stage pipeline that prepares a paired AAVE/SAE dataset of sexually
suggestive content. The output is intended as input to an external
content-moderation tool whose racial bias you want to audit.

## Pipeline

```
TwitterAAE corpus ──▶ [1] AAE filter ──▶ [2] NSFW detector ──▶ [3] AAVE→SAE rewrite ──▶ paired CSV
```

1. **Load** — reads the TwitterAAE TSV (Blodgett et al. 2016), keeps rows with
   high AAE posterior (default `>= 0.8`).
2. **Detect** — runs a local DistilBERT NSFW text classifier (default:
   [`michellejieli/NSFW_text_classifier`](https://huggingface.co/michellejieli/NSFW_text_classifier))
   over the AAVE rows and keeps those scoring above the threshold.
3. **Translate** — rewrites each AAVE row into Standard American English with
   a local LLM via [Ollama](https://ollama.com) (default: `llama3.3:70b`; use
   `--ollama-model llama3.1:8b` on laptops). The prompt instructs the model to
   preserve explicitness so the downstream moderation audit is valid.

Each stage checkpoints to `data/` so you can resume.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install and start Ollama, then pull the translation model:

```bash
brew install ollama          # or download from ollama.com
ollama serve &               # leave running

# On an HPC GPU (≥ ~48 GB VRAM with 4-bit quant), use the 70B model:
ollama pull llama3.3:70b

# On a laptop or smaller GPU, fall back to the 8B model:
ollama pull llama3.1:8b
```

The first run will also download the DistilBERT NSFW classifier (~250 MB) from
HuggingFace into the local cache.

## Usage

Full pipeline on a sample of 5,000 rows:

```bash
python run_pipeline.py --input /path/to/twitteraae_all.tsv --sample 5000
```

Resume from a previous run (skip stages whose checkpoint already exists):

```bash
python run_pipeline.py --input /path/to/twitteraae_all.tsv --skip-existing
```

Run only specific stages (e.g., re-translate with a different Ollama model):

```bash
python run_pipeline.py --stages translate --ollama-model mistral:7b
```

Useful flags:

| Flag | Default | Notes |
|---|---|---|
| `--aae-threshold` | `0.8` | Min posterior probability for the AAE topic |
| `--sample` | (none) | Random subsample after AAE filtering |
| `--nsfw-threshold` | `0.5` | Min NSFW probability to keep a row |
| `--nsfw-model` | `michellejieli/NSFW_text_classifier` | Any HF text-classification model with NSFW/SFW labels |
| `--ollama-model` | `llama3.3:70b` | Any pulled Ollama model. Use `llama3.1:8b` on laptops. |
| `--translate-limit` | (none) | Cap on rows to translate; useful for smoke tests |

## Output

The final artefact is `data/03_paired.csv` with columns:

| column | description |
|---|---|
| `tweet_id` | Original TwitterAAE id |
| `aae_prob` | Posterior probability the tweet is AAE |
| `nsfw_score` | NSFW classifier score (0–1) |
| `aave_message` | Original AAVE text |
| `sae_message` | LLM rewrite in SAE |

Feed `aave_message` and `sae_message` into the external moderation tool you're
auditing and compare flagging rates.

## Notes

- The NSFW classifier is trained on Reddit data and inherits its biases.
  Manually spot-check the flagged rows before drawing conclusions about your
  audit dataset.
- The AAVE→SAE rewriter is an open LLM; rewrites should be spot-checked.
  Even Llama 3.3 70B can flatten dialect features or introduce subtle
  stereotypes (see Hofmann et al. 2024, *Dialect prejudice predicts AI
  decisions about people's character, employability, and criminality*).
  For a publication-quality audit, have rewrites validated by fluent
  AAVE speakers before drawing conclusions about the moderation tool.
