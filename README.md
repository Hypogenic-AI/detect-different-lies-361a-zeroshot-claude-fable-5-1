# Detect different lies: lies vs. honest errors, and what white-box lie detectors detect

**Question.** When an LLM says something false, is it an epistemic failure (it does not know the
answer: a hallucination) or an incentive-driven misreport (it knows the answer and says otherwise: a lie)?
And do standard white-box lie detectors and hallucination detectors tell the two apart, or do they
respond to falsehood, uncertainty, or the prompt?

**Design.** For one model and the same 6,000 TriviaQA questions we cross

- *knowledge*: known / unknown, from neutral elicitation (greedy answer + 10 samples), with
- *incentive*: neutral prompt, five *pressure* prompts (instructed lie, trickster role-play, bluffing
  game, rival team, sandbagging threat) and five matched *control* prompts (same scenario,
  truth-aligned incentive).

Every response lands in a cell: honest correct, hallucination, lie (known + wrong under pressure),
honest under pressure, unknowing falsehood (unknown + wrong under pressure), feigned / genuine
ignorance. Detectors (all scored on mean response-token activations at one layer): an instructed-pairs
deception probe (Goldowsky-Dill et al. recipe), a true/false-statement falsehood probe, an in-domain
instructed-lie probe, a hallucination probe, token NLL, and in-domain reference probes.
Models: Llama-3.1-8B-Instruct and Gemma-2-9B-it.

The paper is in `paper_draft/main.pdf` (source `paper_draft/main.tex`).

## Main findings

| | Llama-3.1-8B | Gemma-2-9B |
|---|---|---|
| Lies on known questions, greedy: instructed / trickster / bluff / rival / sandbag | 39.5 / 4.7 / 1.3 / 0.7 / 0.7 % | 57.1 / 27.3 / 5.9 / 1.3 / 1.4 % |
| Wrong answers on known questions under the matched control prompts | 0.5 - 1.9 % | 0.7 - 1.4 % |
| Feigned ignorance on known questions under the sandbag prompt | 15.4 % | 23.5 % |
| Share of false outputs that are lies: instructed / trickster / bluff / rival / sandbag | 46 / 14 / 4 / 3 / 5 % | 60 / 46 / 15 / 4 / 8 % |
| Deception probe, lie vs honest *control-prompt* answers (instructed) | 1.00 | 1.00 |
| Deception probe, honest under pressure vs honest under control (prompt only) | 1.00 | 1.00 |
| Deception probe, lie vs honest, *same prompt* (instructed / trickster) | 0.75 / 0.78 | 0.64 / 0.68 |
| Falsehood probe, lie vs honest, same prompt (instructed / trickster) | 0.84 / 0.87 | 0.88 / 0.84 |
| Hallucination probe, lie vs honest, same prompt (instructed / trickster) | 0.73 / 0.84 | 0.79 / 0.77 |
| Deception probe, hallucination vs correct (control prompt, instructed scenario) | 0.69 | 0.68 |
| Deception probe, lie vs unknowing falsehood, same prompt (instructed / trickster) | 0.68 / 0.76 | 0.63 / 0.62 |
| Hallucination probe, lie vs unknowing falsehood (instructed) | 0.21 | 0.27 |
| Feigned vs genuine ignorance: best of deception / falsehood / instructed-lie / hallucination probes | 0.59 | 0.62 |
| Feigned vs genuine ignorance: probe trained on that contrast | 0.87 | 0.83 |
| In-domain probe, lie vs all non-lies under the same prompt (instructed) | 0.93 | 0.90 |

(AUROC on held-out questions.) In words:

1. **Without an instruction or a role, these models rarely lie.** Under incentive-only prompts,
   85-97 % of false outputs are epistemic failures. The incentive-driven misreport that does occur is
   feigned ignorance ("I don't know" on a known question).
2. **The usual evaluation measures the prompt.** Deception probes separate lies from honest control
   answers perfectly, and separate honest answers under the two prompts equally perfectly.
3. **With the prompt fixed, detectors respond to falsehood in general.** A truth probe and a
   hallucination probe find lies as well as or better than deception probes; deception probes also rank
   hallucinations above correct answers; no off-the-shelf detector reliably separates a lie from a
   falsehood told without knowledge, or feigned from genuine ignorance.
4. **Sampled "lies" under the bluffing game look like ordinary slips** to uncertainty-type detectors.
5. **The distinction is linearly decodable** by probes trained against same-prompt honest answers and
   same-prompt honest errors.

## Caveats (see the Limitations section of the paper)

- Two small models (8-9B); a planned third (Qwen2.5-14B) was not run for compute-time reasons.
  Non-instructed lies were rare, so incentive-driven results rest on role-play, a sampled game
  incentive (a quarter to a half of those "lies" are expected to be slips) and feigned ignorance.
- **Judge.** The plan was to grade with GPT-4.1-mini via OpenRouter, but the key hit its daily limit
  part-way through the first model (and the provided OpenAI key was rejected). All labels in the paper
  therefore come from a local judge (Qwen2.5-14B-Instruct, next-token logits over four labels). It
  agrees with the GPT-4.1-mini grades that had been collected on 92.9 % of 9,473 responses
  (`results/judge_validation_qwen14.json`); correct/wrong swaps are under 1 %.
- Knowledge labels come from sampling consistency under one neutral prompt; "middle" questions
  (14-26 %) are excluded from detector analyses.
- One probe family, one layer per model (chosen on train-split questions among five candidates).

## Layout

```
src/common.py         prompts (SCENARIOS), model loading, batched generation + activation capture
src/run_model.py      phase 1: all prompts, greedy, activations; neutral samples; probe training sets
src/local_judge.py    local judge (Qwen2.5-14B); run as a script to validate against cached GPT grades
src/judge.py          API judge (GPT-4.1-mini via OpenRouter); used for the pilot and the validation set
src/label.py          grade phase-1 responses, derive knowledge labels
src/run_samples.py    phase 2: temperature-1 samples under bluff (pressure, control) and trickster prompts
src/label_samples.py  grade phase-2 samples
src/analyze.py        accounting, probes, AUROC contrasts, fire rates, prompt-token checks, paired analysis
src/make_figs.py      figures and LaTeX tables for the paper
src/pilot.py          prompt pilot (judge-graded lie yield on a few hundred questions)
run_all.sh            the full pipeline per model
results/<model>/      accounting.json, auroc.json, layer_sweep.json, fire.json, lastprompt.json,
                      paired.json, bluff_strict.json, examples.json, scores_mid.npz (per-response scores),
                      responses/ (all generated responses, judge labels, knowledge labels)
paper_draft/          main.tex, sections/, tables/ (generated), figs/ (generated), main.pdf
```

Activations (about 45 GB), model weights and caches live under `data/`, `models/` and `.cache/`
and are git-ignored.

## Reproduce

```bash
uv venv .venv --python 3.12
uv pip install -p .venv/bin/python "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu126
uv pip install -p .venv/bin/python transformers accelerate datasets scikit-learn scipy numpy pandas \
    matplotlib openai tqdm huggingface_hub pymupdf
export HF_TOKEN=...            # gated: meta-llama/Llama-3.1-8B-Instruct, google/gemma-2-9b-it
# probe training statements
mkdir -p data && cd data
curl -sLO https://raw.githubusercontent.com/ApolloResearch/deception-detection/main/data/repe/true_false_facts.csv
for f in cities sp_en_trans companies_true_false common_claim_true_false; do
  curl -sL -o got_$f.csv https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets/$f.csv; done
cd ..
./run_all.sh llama gemma       # about 6-7 h per model on one 48 GB GPU
cd src && ../.venv/bin/python make_figs.py
cd ../paper_draft && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`run_all.sh` starts `analyze.py` in the background after each model; wait for
`logs/analyze_<model>.log` to print `analysis done` before running `make_figs.py`.
The judge-validation script (`python local_judge.py qwen14`) additionally needs the cached
GPT-4.1-mini grades (`data/judge_cache.sqlite`, not regenerated without an OpenRouter key).
Sampling at temperature 1 is not seeded, so regenerated numbers will differ slightly.
