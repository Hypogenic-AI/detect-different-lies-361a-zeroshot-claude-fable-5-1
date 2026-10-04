"""Grade all phase-1 responses with the local judge and derive per-question knowledge labels."""
import sys
from collections import Counter
from common import *
from local_judge import load_judge, grade, agree

key = sys.argv[1]
OUT = os.path.join(ROOT, "data", "runs", key)
if os.path.exists(f"{OUT}/labels.json"):
    sys.exit(0)
qs = jload(f"{OUT}/questions.json")
conds = list(conditions())
rec = {c: jload(f"{OUT}/{c}.json") for c in conds}
samples = jload(f"{OUT}/neutral_samples.json")
judge = load_judge()
NS = len(samples[0])
flat = [(q, s) for q, ss in zip(qs, samples) for s in ss]
S = np.array(grade(judge, flat, desc="neutral samples")).reshape(len(qs), NS)
G = {c: grade(judge, [(q, r["response"]) for q, r in zip(qs, rec[c])], desc=c) for c in conds}

labels = []
for i, q in enumerate(qs):
    g0 = G["neutral"][i]
    n_correct = int((S[i] == "CORRECT").sum())
    if g0 == "CORRECT" and n_correct >= NS - 1:
        k = "known"       # belief = reference answer
    elif g0 != "CORRECT" and n_correct <= 1:
        k = "unknown"     # the model does not have the answer
    else:
        k = "middle"
    r0 = rec["neutral"][i]["response"]
    # how many neutral samples repeat the greedy (wrong) answer: a consistent wrong belief vs. a dispersed guess
    same = sum(agree(r0, s, q["question"]) for s, l in zip(samples[i], S[i]) if l == "WRONG") if g0 == "WRONG" else 0
    d = {"qid": q["qid"], "knowledge": k, "n_correct": n_correct, "top_wrong_share": same / NS,
         "n_abstain": int((S[i] == "ABSTAIN").sum()), "sample_labels": S[i].tolist()}
    for c in conds:
        d[c] = {"label": G[c][i]}
    labels.append(d)
jdump(labels, f"{OUT}/labels.json")
print(Counter(l["knowledge"] for l in labels))
for c in conds:
    for k in ["known", "unknown", "middle"]:
        print(c, k, dict(Counter(l[c]["label"] for l in labels if l["knowledge"] == k)))
