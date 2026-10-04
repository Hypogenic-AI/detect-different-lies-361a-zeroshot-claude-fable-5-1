"""Figures and LaTeX tables for the paper, from results/<model>/*.json."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import *

MODELS_RUN = [m for m in ["llama", "gemma", "qwen14"] if os.path.exists(f"{ROOT}/results/{m}/auroc.json")]
NAME = {"llama": "Llama-3.1-8B", "gemma": "Gemma-2-9B", "qwen14": "Qwen2.5-14B"}
FIG = f"{ROOT}/paper_draft/figs"
TAB = f"{ROOT}/paper_draft/tables"
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
R = {m: {k: jload(f"{ROOT}/results/{m}/{k}.json") for k in ["accounting", "auroc", "layer_sweep", "fire", "lastprompt", "paired", "bluff_strict"]}
     for m in MODELS_RUN}
SCEN = list(SCENARIOS)
DET = ["repe", "truth", "lie_instr", "halluc", "nll", "combo"]
DNAME = {"repe": "Deception probe (instructed-pairs)", "truth": "Falsehood probe (true/false statements)",
         "lie_instr": "Lie probe (instructed lies, in-domain)", "halluc": "Hallucination probe (neutral QA)",
         "nll": "Token NLL (uncertainty)", "combo": "Falsehood minus hallucination"}
DSHORT = {"repe": "Deception (IP)", "truth": "Falsehood", "lie_instr": "Instr.-lie", "halluc": "Halluc.", "nll": "NLL", "combo": "False$-$Halluc."}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
C = {"known": "#c0392b", "middle": "#bdbdbd", "unknown_consistent": "#1f5f99", "unknown_dispersed": "#7fb3e0"}


def f2(x):
    return "--" if x is None else f"{(x[0] if isinstance(x, list) else x):.2f}"


def fci(x):
    return "--" if x is None else f"{x[0]:.2f}\\,{{\\scriptsize[{x[1]:.2f},{x[2]:.2f}]}}"


# ---------------------------------------------------------------- Fig 1: composition of false outputs
conds = ["neutral"] + [f"{s}_pressure" for s in SCEN]
cl = ["neutral"] + SCEN
fig, axs = plt.subplots(1, len(MODELS_RUN), figsize=(2.3 * len(MODELS_RUN) + 0.6, 2.3), sharey=True, squeeze=False)
for ax, m in zip(axs[0], MODELS_RUN):
    A = R[m]["accounting"]
    bottom = np.zeros(len(conds))
    for part, lab in [("known", "model knows the answer (lie / slip)"), ("middle", "inconsistent knowledge"),
                      ("unknown_consistent", "does not know, consistent wrong answer"),
                      ("unknown_dispersed", "does not know, dispersed guesses")]:
        v = np.array([A[c]["false_share"][part] for c in conds])
        ax.bar(range(len(conds)), v, bottom=bottom, color=C[part], label=lab, width=0.75)
        bottom += v
    for i, c in enumerate(conds):
        ax.text(i, 1.01, str(A[c]["false_share"]["n_false"]), ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(cl, rotation=40, ha="right")
    ax.set_title(NAME[m], pad=10)
    ax.set_ylim(0, 1.0)
axs[0][0].set_ylabel("share of false outputs")
axs[0][0].legend(loc="upper center", bbox_to_anchor=(len(MODELS_RUN) / 2, -0.38), ncol=2, frameon=False, fontsize=7)
plt.savefig(f"{FIG}/composition.pdf", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------- Table: behaviour on known questions
rows = []
for m in MODELS_RUN:
    A = R[m]["accounting"]
    nk = A["knowledge"]["known"]
    for s in SCEN:
        p, c = A[f"{s}_pressure"]["known"], A[f"{s}_control"]["known"]
        rows.append(f"{NAME[m] if s == SCEN[0] else ''} & {s} & " + " & ".join(
            f"{100 * p.get(l, 0) / nk:.1f}" for l in ["CORRECT", "WRONG", "ABSTAIN", "MIXED"])
            + f" & {100 * c.get('WRONG', 0) / nk:.1f} & {100 * c.get('ABSTAIN', 0) / nk:.1f} \\\\")
    rows[-1] += " \\midrule"
open(f"{TAB}/behaviour.tex", "w").write("\n".join(rows))

rows = []
for m in MODELS_RUN:
    A = R[m]["accounting"]
    k = A["knowledge"]
    rows.append(f"{NAME[m]} & {k['known']} & {k['middle']} & {k['unknown']} & {A['unknown_consistent']} \\\\")
open(f"{TAB}/knowledge.tex", "w").write("\n".join(rows))

rows = []
for m in MODELS_RUN:
    S = R[m]["accounting"].get("samples", {})
    for c in S:
        kn = S[c]["known"]
        tot = sum(kn.values())
        rows.append(f"{NAME[m]} & {c.replace('_', ' ')} & " + " & ".join(f"{100 * kn.get(l, 0) / tot:.1f}" for l in ["CORRECT", "WRONG", "ABSTAIN", "MIXED"])
                    + f" & {100 * S[c]['known_questions_with_wrong'] / S[c]['known_questions']:.1f} \\\\")
open(f"{TAB}/samples.tex", "w").write("\n".join(rows))

# ---------------------------------------------------------------- Table: main AUROC matrix
CON = [("lie_vs_honest_control", "lie vs honest (control prompt)"), ("lie_vs_honest_same", "lie vs honest (same prompt)"),
       ("prompt_only", "honest: pressure vs control prompt"), ("halluc_vs_correct_control", "hallucination vs correct (control)"),
       ("lie_vs_unknowing_same", "lie vs unknowing falsehood (same prompt)"),
       ("lie_vs_halluc_control", "lie vs hallucination (control)"),
       ("unknowing_vs_honest_same", "unknowing falsehood vs honest (same prompt)"),
       ("lie_vs_rest_same", "lie vs all non-lies (same prompt)")]
for m in MODELS_RUN:
    T, n = R[m]["auroc"]["auroc"], R[m]["auroc"]["n"]
    rows = []
    for s in ["instructed", "trickster", "bluff"]:
        rows.append(f"\\multicolumn{{{len(DET) + 2}}}{{l}}{{\\emph{{{s}}}}} \\\\")
        for cname, cl_ in CON:
            nn = n[s][cname]
            rows.append(f"\\;\\;{cl_} & {nn[0]}/{nn[1]} & " + " & ".join(f2(T[d][s][cname]) if d in T else "--" for d in DET) + " \\\\")
    open(f"{TAB}/auroc_{m}.tex", "w").write("\n".join(rows))
    # CI version of the key contrasts
    rows = []
    for s in ["instructed", "trickster", "bluff"]:
        for cname in ["lie_vs_honest_same", "lie_vs_unknowing_same", "lie_vs_rest_same", "halluc_vs_correct_control"]:
            rows.append(f"{s} & {cname.replace('_', ' ')} & " + " & ".join(fci(T[d][s][cname]) if d in T else "--" for d in DET) + " \\\\")
    open(f"{TAB}/auroc_ci_{m}.tex", "w").write("\n".join(rows))

# ---------------------------------------------------------------- Fig 2: AUROC heatmaps (all models)
fig, axs = plt.subplots(len(MODELS_RUN), 3, figsize=(7.2, 1.75 * len(MODELS_RUN) + 0.9), squeeze=False)
for r, m in enumerate(MODELS_RUN):
    T = R[m]["auroc"]["auroc"]
    for k, s in enumerate(["instructed", "trickster", "bluff"]):
        ax = axs[r][k]
        M = np.array([[np.nan if (d not in T or T[d][s][c] is None) else T[d][s][c][0] for c, _ in CON] for d in DET])
        im = ax.imshow(M, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}"[1:] if M[i, j] < 0.995 else "1.0", ha="center", va="center", fontsize=6.5,
                            color="white" if abs(M[i, j] - 0.5) > 0.3 else "black")
        ax.set_xticks(range(len(CON)))
        ax.set_xticklabels([f"C{j + 1}" for j in range(len(CON))] if r < len(MODELS_RUN) - 1 else
                           [f"C{j + 1}: {c[1]}" for j, c in enumerate(CON)], rotation=0 if r < len(MODELS_RUN) - 1 else 55,
                           ha="center" if r < len(MODELS_RUN) - 1 else "right", fontsize=6.5)
        ax.set_yticks(range(len(DET)))
        ax.set_yticklabels([DSHORT[d] for d in DET] if k == 0 else [])
        ax.set_title(f"{NAME[m]} / {s}", fontsize=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
plt.subplots_adjust(wspace=0.08, hspace=0.45)
plt.savefig(f"{FIG}/auroc_heatmap.pdf", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------- Fig 3: score distributions per cell
CELLS = [("control correct", "honest correct\n(control)", "#4d9d5b"), ("control hallucination", "hallucination\n(control)", "#1f5f99"),
         ("pressure honest", "honest\n(pressure)", "#9bd0a4"), ("pressure lie", "lie\n(pressure)", "#c0392b"),
         ("pressure unknowing falsehood", "unknowing falsehood\n(pressure)", "#8e6bb0")]
for m in MODELS_RUN:
    F = R[m]["fire"]
    dets = [d for d in ["repe", "truth", "lie_instr", "halluc"] if d in F]
    fig, axs = plt.subplots(len(dets), 3, figsize=(7.0, 1.25 * len(dets) + 0.5), squeeze=False)
    for i, d in enumerate(dets):
        ref = F[d]["neutral|correct"]["q"]
        sd = (ref[3] - ref[1]) / 1.349 or 1.0
        for k, s in enumerate(["instructed", "trickster", "bluff"]):
            ax = axs[i][k]
            for j, (cn, lab, col) in enumerate(CELLS):
                e = F[d][f"{s}|{cn}"]
                if e["q"] is None or e["n"] < 20:
                    continue
                q = [(v - ref[2]) / sd for v in e["q"]]
                ax.plot([j, j], [q[0], q[4]], color=col, lw=1)
                ax.add_patch(plt.Rectangle((j - 0.3, q[1]), 0.6, q[3] - q[1], color=col, alpha=0.85))
                ax.plot([j - 0.3, j + 0.3], [q[2], q[2]], color="white", lw=1)
                ax.text(j, q[4], f"{100 * e['fire']:.0f}%", ha="center", va="bottom", fontsize=5.5)
            ax.axhline(0, color="gray", lw=0.5, ls=":")
            ax.set_xticks(range(len(CELLS)))
            ax.set_xticklabels([c[1] for c in CELLS] if i == len(dets) - 1 else [], fontsize=5.5, rotation=30, ha="right")
            ax.set_xlim(-0.6, len(CELLS) - 0.4)
            if k == 0:
                ax.set_ylabel(DSHORT[d], fontsize=7)
            if i == 0:
                ax.set_title(s, fontsize=8)
    plt.subplots_adjust(wspace=0.18, hspace=0.25)
    plt.savefig(f"{FIG}/cells_{m}.pdf", bbox_inches="tight")
    plt.close()

# ---------------------------------------------------------------- Table: fire rates at 1% FPR
for m in MODELS_RUN:
    F = R[m]["fire"]
    rows = []
    for s in ["instructed", "trickster", "bluff", "sandbag"]:
        for cn in ["control correct", "control hallucination", "pressure honest", "pressure lie",
                   "pressure unknowing falsehood", "pressure feigned ignorance", "pressure genuine ignorance"]:
            e = F["repe"][f"{s}|{cn}"]
            if e["n"] < 20:
                continue
            rows.append(f"{s} & {cn} & {e['n']} & " + " & ".join(f"{100 * F[d][f'{s}|{cn}']['fire']:.0f}" if d in F else "--" for d in DET) + " \\\\")
        rows[-1] += " \\midrule"
    open(f"{TAB}/fire_{m}.tex", "w").write("\n".join(rows))

# ---------------------------------------------------------------- Fig 4: layer sweep
fig, axs = plt.subplots(len(MODELS_RUN), 4, figsize=(7.2, 1.55 * len(MODELS_RUN) + 0.4), squeeze=False, sharey=True)
PAN = [("instructed", "lie_vs_honest_same", "instructed: lie vs honest\n(same prompt)"),
       ("instructed", "lie_vs_unknowing_same", "instructed: lie vs\nunknowing falsehood"),
       ("trickster", "lie_vs_honest_same", "trickster: lie vs honest\n(same prompt)"),
       ("instructed", "halluc_vs_correct_control", "hallucination vs correct\n(control prompt)")]
DC = {"repe": "#c0392b", "truth": "#e08e0b", "lie_instr": "#8e6bb0", "halluc": "#1f5f99"}
for r, m in enumerate(MODELS_RUN):
    W = R[m]["layer_sweep"]
    Ls = sorted(int(l) for l in W)
    for k, (s, cn, title) in enumerate(PAN):
        ax = axs[r][k]
        for d in ["repe", "truth", "lie_instr", "halluc"]:
            y = [W[str(l)].get(d, {}).get(s, {}).get(cn) for l in Ls]
            y = [np.nan if v is None else v for v in y]
            ax.plot(Ls, y, color=DC[d], label=DSHORT[d], lw=1.2)
        ax.axhline(0.5, color="gray", lw=0.5, ls=":")
        ax.axvline(R[m]["auroc"]["layer"], color="gray", lw=0.5)
        ax.set_ylim(0, 1)
        if r == 0:
            ax.set_title(title, fontsize=7)
        if k == 0:
            ax.set_ylabel(f"{NAME[m]}\nAUROC", fontsize=7)
        if r == len(MODELS_RUN) - 1:
            ax.set_xlabel("layer")
axs[-1][1].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(1.05, -0.38), ncol=4)
plt.subplots_adjust(wspace=0.1, hspace=0.3)
plt.savefig(f"{FIG}/layers.pdf", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------- Table: prompt-token checks, feigned ignorance, paired
rows = []
for m in MODELS_RUN:
    LP = R[m]["lastprompt"]
    for d in ["repe", "lie_instr"]:
        if d not in LP["prompt_identity_resp"]:
            continue
        rows.append(f"{NAME[m]} & {DSHORT[d]} & response & " + " & ".join(f2(LP["prompt_identity_resp"][d][s]) for s in SCEN) + " \\\\")
        rows.append(f" & {DSHORT[d]} & last prompt token & " + " & ".join(f2(LP["prompt_identity_last"][d][s]) for s in SCEN) + " \\\\")
    rows[-1] += " \\midrule"
open(f"{TAB}/prompt_identity.tex", "w").write("\n".join(rows))

rows = []
for m in MODELS_RUN:
    LT = R[m]["lastprompt"]["trained_on_last_token"]
    for d, dl in [("lie_instr", "instr.-lie probe"), ("same:instructed", "same-prompt lie probe (instructed)"),
                  ("same:trickster", "same-prompt lie probe (trickster)"), ("halluc", "hallucination probe")]:
        if d not in LT:
            continue
        s = d.split(":")[1] if ":" in d else "instructed"
        rows.append(f"{NAME[m]} & {dl} & " + " & ".join(f2(LT[d][s][c]) for c in ["lie_vs_honest_control", "lie_vs_honest_same", "prompt_only",
                                                                                 "lie_vs_unknowing_same", "halluc_vs_correct_control"]) + " \\\\")
    rows[-1] += " \\midrule"
open(f"{TAB}/last_token.tex", "w").write("\n".join(rows))

rows = []
for m in MODELS_RUN:
    T, n = R[m]["auroc"]["auroc"], R[m]["auroc"]["n"]
    for cn, cl_ in [("feigned_vs_genuine_ignorance", "feigned vs genuine ignorance"), ("feigned_vs_honest_same", "feigned ignorance vs honest answer")]:
        nn = n["sandbag"][cn]
        rows.append(f"{NAME[m] if cn.startswith('feigned_vs_g') else ''} & {cl_} & {nn[0]}/{nn[1]} & " + " & ".join(
            fci(T[d]["sandbag"][cn]) if d in T else "--" for d in DET + ["feign"]) + " \\\\")
    rows[-1] += " \\midrule"
open(f"{TAB}/feigned.tex", "w").write("\n".join(rows))

rows = []
for m in MODELS_RUN:
    Pd = R[m]["paired"]
    for c in Pd:
        e = Pd[c]
        rows.append(f"{NAME[m]} & {c.split('_')[0]} & {e['n_mixed']} & " + " & ".join(
            (f"{e['det'][d]['paired_win']:.2f}\\,{{\\scriptsize[{e['det'][d]['ci'][0]:.2f},{e['det'][d]['ci'][1]:.2f}]}}" if d in e["det"] and e["det"][d]["paired_win"] is not None else "--")
            for d in DET) + " \\\\")
open(f"{TAB}/paired.tex", "w").write("\n".join(rows))

# in-domain reference probes
rows = []
for m in MODELS_RUN:
    T = R[m]["auroc"]["auroc"]
    g = lambda d, s, c: T.get(d, {}).get(s, {}).get(c)
    for s in ["instructed", "trickster", "bluff"]:
        rows.append(f"{NAME[m] if s == 'instructed' else ''} & {s} & {f2(g(f'same:{s}', s, 'lie_vs_honest_same'))} & {f2(g(f'lvu:{s}', s, 'lie_vs_unknowing_same'))} & "
                    f"{f2(g(f'ovr:{s}', s, 'lie_vs_rest_same'))} & {f2(g('ovr:instructed', s, 'lie_vs_rest_same'))} & "
                    f"{f2(g(f'ovr:{s}', s, 'halluc_vs_correct_control'))} & {f2(g(f'lvu:{s}', s, 'halluc_vs_correct_control'))} \\\\")
    rows[-1] += " \\midrule"
open(f"{TAB}/indomain.tex", "w").write("\n".join(rows))
print("figs done", MODELS_RUN)

# ---------------------------------------------------------------- Appendix material: prompts, judge validation, examples
def tex(s):
    for a, b in [("\\", "\\textbackslash "), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}")]:
        s = s.replace(a, b)
    return s.encode("ascii", "ignore").decode()


rows = []
for s, d in SCENARIOS.items():
    rows.append(f"\\paragraph{{{s.capitalize()}, pressure.}} {tex(d['pressure'])}")
    rows.append(f"\\paragraph{{{s.capitalize()}, control.}} {tex(d['control'])}")
open(f"{TAB}/prompts.tex", "w").write("\n\n".join(rows))

jv = jload(f"{ROOT}/results/judge_validation_qwen14.json")
names = ["CORRECT", "WRONG", "ABSTAIN", "MIXED"]
open(f"{TAB}/judge.tex", "w").write("\n".join(
    f"{a.lower()} & " + " & ".join(str(jv["confusion_gpt_rows"][a][b]) for b in names) + " \\\\" for a in names))

for m in MODELS_RUN:
    ex = jload(f"{ROOT}/results/{m}/examples.json")
    rows = []
    for k in ["instructed/lie", "trickster/lie", "bluff/lie", "sandbag/feigned", "instructed/unknowing"]:
        for x in ex.get(k, [])[:2]:
            rows.append(f"{k.replace('/', ' / ')} & {tex(x['q'][:110])} & {tex(x['gold'][:30])} & {tex(x['neutral'][:90])} & {tex(x['pressure'][:110])} \\\\")
    open(f"{TAB}/examples_{m}.tex", "w").write("\n".join(rows))

# ---------------------------------------------------------------- Table: effect decomposition (z units of neutral honest-correct scores)
for m in MODELS_RUN:
    F = R[m]["fire"]
    rows = []
    for s in ["instructed", "trickster", "bluff"]:
        for lab, a, b in [("prompt (honest, pressure $-$ control)", "pressure honest", "control correct"),
                          ("hallucination (control, halluc.\\ $-$ correct)", "control hallucination", "control correct"),
                          ("lie (pressure, lie $-$ honest)", "pressure lie", "pressure honest"),
                          ("unknowing falsehood (pressure, $-$ honest)", "pressure unknowing falsehood", "pressure honest")]:
            rows.append(f"{s} & {lab} & " + " & ".join(
                f"{F[d][f'{s}|{a}']['z'] - F[d][f'{s}|{b}']['z']:+.1f}" if d in F else "--" for d in DET[:5]) + " \\\\")
        rows[-1] += " \\midrule"
    open(f"{TAB}/effects_{m}.tex", "w").write("\n".join(rows))

# ---------------------------------------------------------------- Table: composition of false outputs incl. controls
rows = []
for m in MODELS_RUN:
    A = R[m]["accounting"]
    for c in ["neutral"] + [f"{s}_{k}" for s in SCEN for k in ("pressure", "control")]:
        fs = A[c]["false_share"]
        sn = A[c].get("unknown_wrong_same_as_neutral")
        rows.append(f"{NAME[m] if c == 'neutral' else ''} & {c.replace('_', ' ')} & {fs['n_false']} & {100 * fs['known']:.1f} & {100 * fs['middle']:.1f} & "
                    f"{100 * fs['unknown_consistent']:.1f} & {100 * fs['unknown_dispersed']:.1f} & " + (f"{100 * sn[0] / max(sn[1], 1):.0f}" if sn else "--") + " \\\\")
    rows[-1] += " \\midrule"
open(f"{TAB}/composition.tex", "w").write("\n".join(rows))

# ---------------------------------------------------------------- Table: bluff lies vs slips
rows = []
for m in MODELS_RUN:
    B = R[m]["bluff_strict"]
    for cn, cl_ in [("strict_lie_vs_honest_same", "wrong (clean question) vs honest, bluff pressure"),
                    ("other_wrong_vs_honest_same", "wrong (slip-prone question) vs honest, bluff pressure"),
                    ("control_slip_vs_correct", "wrong vs correct, bluff control (no incentive)")]:
        rows.append(f"{NAME[m] if cn.startswith('strict') else ''} & {cl_} & " + " & ".join(fci(B["det"][d][cn]) if d in B["det"] else "--" for d in DET[:5]) + " \\\\")
    rows[-1] += " \\midrule"
open(f"{TAB}/bluff_strict.tex", "w").write("\n".join(rows))

# strip trailing midrule from generated tables (they are followed by \\bottomrule)
import glob
for fn in glob.glob(f"{TAB}/*.tex"):
    s = open(fn).read().rstrip()
    if s.endswith("\\midrule"):
        open(fn, "w").write(s[: -len("\\midrule")].rstrip() + "\n")
