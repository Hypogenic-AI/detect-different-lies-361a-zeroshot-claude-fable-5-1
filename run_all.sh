#!/bin/bash
# Full pipeline for the given models (default: llama gemma). Requires HF_TOKEN and OPENROUTER_KEY.
cd "$(dirname "$0")/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../.venv/bin/python
for m in ${@:-llama gemma}; do
  $PY run_model.py $m 6000 > ../logs/run_$m.log 2>&1      # phase 1: generation + activations
  $PY label.py $m > ../logs/label_$m.log 2>&1             # LLM-judge grading + knowledge labels
  $PY run_samples.py $m > ../logs/samples_$m.log 2>&1     # phase 2: sampled pressure responses
  $PY label_samples.py $m > ../logs/label_samples_$m.log 2>&1
  (nohup $PY analyze.py $m > ../logs/analyze_$m.log 2>&1 &)
done
echo ALLDONE > ../logs/alldone
