"""Phase 2: for questions the model knows or does not know, sample several responses (T=1) under the
bluff (pressure and control) and trickster prompts. Greedy decoding yields very few non-instructed lies, so the sampled
pool is the main source of incentive-driven lies; it also gives within-question matched lie / honest pairs."""
import sys
from common import *

key = sys.argv[1]
NS = 8
OUT = os.path.join(ROOT, "data", "runs", key)
tok, model = load_model(key)
qs = jload(f"{OUT}/questions.json")
labels = jload(f"{OUT}/labels.json")
L_all = jload(f"{OUT}/meta.json")["layers"]
n = model.config.num_hidden_layers
mid = (n // 4) * 2
L = [l for l in L_all if l in (mid - 8, mid - 2, mid, mid + 2, mid + 8)]
sel = [i for i, l in enumerate(labels) if l["knowledge"] in ("known", "unknown")]
SAMPLE_CONDS = ["bluff_pressure", "bluff_control", "trickster_pressure"]
jdump({"qidx": sel, "layers": L, "n": NS, "conds": SAMPLE_CONDS}, f"{OUT}/samples_meta.json")
for c in SAMPLE_CONDS:
    if os.path.exists(f"{OUT}/samples_{c}.json"):
        continue
    prompts = [build_chat(tok, conditions()[c], USER_TMPL.format(q=qs[i]["question"])) for i in sel]
    rec, A, _ = generate(tok, model, prompts, bs=12, do_sample=True, n=NS, capture=True, layers=L, desc=c)
    np.save(f"{OUT}/samples_{c}_resp.npy", A)
    jdump(rec, f"{OUT}/samples_{c}.json")
print("done")
