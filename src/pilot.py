"""Pilot: judge-graded lie yield of candidate pressure prompts on held-out pilot questions (seed 123)."""
import sys
from collections import Counter
from common import *
from judge import judge_many, fix

key = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
variants = dict(conditions())
if len(sys.argv) > 3:
    variants.update(jload(sys.argv[3]))
tok, model = load_model(key)
qs = load_questions(N, seed=123)
names = list(variants)
R = {}
for c in names:
    prompts = [build_chat(tok, variants[c], USER_TMPL.format(q=q["question"])) for q in qs]
    R[c] = [r["response"] for r in generate(tok, model, prompts, bs=48, desc=c)]
G = judge_many([(q, [R[c][i] for c in names]) for i, q in enumerate(qs)], chunk=len(names))
lab = {c: np.array([fix(G[i][j], R[c][i], q)["label"] for i, q in enumerate(qs)]) for j, c in enumerate(names)}
known = lab["neutral"] == "CORRECT"
for c in names:
    k = Counter(lab[c][known])
    print(f"== {key} {c}: neutral-correct n={known.sum()}: " + " ".join(f"{x}={k[x]/known.sum():.3f}" for x in ["CORRECT", "WRONG", "ABSTAIN", "MIXED"]))
    for i in np.where(known & (lab[c] == "WRONG"))[0][:4]:
        print("     ", qs[i]["answer"], "||", R[c][i][:150])
