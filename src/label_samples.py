"""Grade the phase-2 sampled responses with the local judge."""
import sys
from collections import Counter
from common import *
from local_judge import load_judge, grade

key = sys.argv[1]
OUT = os.path.join(ROOT, "data", "runs", key)
qs = jload(f"{OUT}/questions.json")
m = jload(f"{OUT}/samples_meta.json")
judge = load_judge()
for s in m["conds"]:
    rec = jload(f"{OUT}/samples_{s}.json")
    flat = [(qs[qi], rec[j * m["n"] + k]["response"]) for j, qi in enumerate(m["qidx"]) for k in range(m["n"])]
    out = np.array(grade(judge, flat, desc=s)).reshape(len(m["qidx"]), m["n"]).tolist()
    jdump(out, f"{OUT}/samples_{s}_labels.json")
    print(s, Counter(x for o in out for x in o), "mixed questions:", sum(("WRONG" in o and "CORRECT" in o) for o in out))
