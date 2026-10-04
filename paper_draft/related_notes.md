# Related-work notes (verified 2026-10-03)

How these were verified: title, authors and first-submission date were pulled from the arXiv API
record for each ID (the same record shown at `https://arxiv.org/abs/ID`). Three ACL papers were also
checked against the ACL Anthology `.bib`; the Nature and Patterns papers against their Crossref DOI
records; the Anthropic post against the page itself. Summaries are based on the **abstracts only**
(plus the body of the Anthropic blog post) -- full PDFs were not read, so method details beyond the
abstract are not confirmed.

## Things to fix in the draft

- `2510.09033`: the current arXiv title (v3) is "Do LLMs Really Know What They Don't Know? Internal
  States Mainly Reflect Knowledge Recall Rather Than Truthfulness", not "LLMs Do NOT Really Know
  What They Don't Know". The earlier-version title was not checked.
- Simhi et al. `2502.12964`: the current arXiv title (v2) is "Trust Me, I'm Wrong: LLMs Hallucinate
  with Certainty Despite Knowing the Answer", not "...High-Certainty Hallucinations in LLMs".
- `2603.10003` has an arXiv ID from March 2026 but its v1 submission date is 2026-02-16. Year 2026
  either way.
- Venue notes marked "per arXiv comment field" in the .bib (COLM 2024 for Geometry of Truth, ICLR
  2023 for CCS and Semantic Uncertainty, NeurIPS 2024 for Truth is Universal, COLM 2025 for
  Inside-Out, TOIS for Huang et al., ACM CSUR for Ji et al.) are the authors' own statements on
  arXiv; proceedings pages were not checked, so they are kept as `@misc` with a note.
- Orgad et al. and Kossen et al. are cited as arXiv preprints; no venue was verified for them.

## Could not be verified

None of the requested items failed. Every listed arXiv ID resolved, and every named paper was found.
Two caveats:
- "Yuan/others, mechanistic approach to sandbagging/lying": no such paper was searched for or found;
  the example given instead (Campbell et al., "Localizing Lying in Llama") was verified and included.
- Smith et al. (2025), cited inside the Anthropic post as arguing that strategic deception is hard
  to distinguish from reflexive responses, looks relevant but was NOT looked up and is NOT in the
  .bib.

## Overlap with the knowledge x incentive design

From the abstracts, no listed paper crosses "model knows / does not know" with "incentive / no
incentive" and then tests whether deception probes and hallucination probes separate the cells.
The nearest neighbours, in order of closeness:

1. `ren2025mask` -- elicits beliefs neutrally, then applies pressure, and scores honesty (statement
   vs. belief) separately from accuracy (belief vs. truth). This is the same conceptual split, but
   behavioural: no detection probes (only a representation-engineering intervention to improve
   honesty).
2. `hopkins2026finetuned` -- same belief-elicitation-then-pressure pipeline (partly drawn from
   MASK), defines a lie as false + known-false, and notes about 25% of labels changed after a judge
   pass because deception is hard to tell from confusion. Detectors are fine-tuned/prompted models
   answering "did you lie?", not activation probes, and hallucination is not a separate condition.
3. `cheang2025really` -- on the hallucination side only: hidden states track whether the model is
   recalling parametric knowledge rather than whether the output is true. Directly relevant to why
   a "hallucination probe" may really be a knowledge-recall probe. No incentive manipulation.
4. `simhi2024distinguishing`, `simhi2025trust` -- split hallucinations by whether the model holds
   the knowledge (HK- vs HK+ / CHOKE). This is the knowledge axis without the incentive axis.
5. `campbell2023localizing` -- the abstract states the motivating ambiguity explicitly (false
   outputs from lack of knowledge vs. dishonesty) but studies only instructed lying.
6. `kretschmar2025liarsbench`, `cooney2026didyoulie` -- vary the reason for lying and verify
   beliefs, but all negatives are honest responses, not hallucinations.

This is a reading of abstracts; a full-text check of MASK, Liars' Bench, Cooney et al. and Cheang et
al. is advisable before claiming novelty.

## Per-paper notes

### Lying / deception benchmarks and probes

**`ren2025mask`** (arXiv 2503.03750) -- Argues that many "honesty" benchmarks really measure
accuracy, and introduces a large human-collected dataset for measuring lying directly, so honesty
and accuracy can be disentangled. Larger models are more accurate but not more honest, and most
frontier models lie substantially under pressure. Representation-engineering interventions improve
honesty. Core source for the lie (knowing misreport under pressure) vs. inaccuracy distinction.

**`kretschmar2025liarsbench`** (arXiv 2511.16035) -- Testbed of 72,863 lies and honest responses
from four open-weight models across seven datasets, varying the model's reason for lying and the
object of belief. Defines lying as generating statements the model believes are false. Three black-
and white-box detectors systematically fail on some lie types, especially where the lie cannot be
determined from the transcript alone.

**`goldowskydill2025detecting`** (arXiv 2502.03407) -- Trains linear probes on Llama-3.3-70B-Instruct
activations using contrasting honest/deceptive instructions (following Zou et al.) and roleplay
scenarios, then tests on insider-trading concealment and sandbagging. AUROC 0.96-0.999; at 1% FPR on
unrelated chat data the probe catches 95-99% of deceptive responses. Authors judge this insufficient
as a robust defence. The standard white-box deception probe baseline.

**`natarajan2026oneprobe`** (arXiv 2602.01425; ICML 2026 per arXiv journal-ref) -- Shows deception
detection is heterogeneous: a universal probe gives +0.032 AUC, while probes matched to deception
type give +0.108 AUC in oracle analysis. Instruction pairs capture deceptive intent rather than
content, and prompt choice explains 70.6% of probe-performance variance. Recommends threat-model-
specific probes over a universal detector.

**`thormann2026probing`** (arXiv 2603.10003) -- Challenges the assumption that deception equals
lying. Across three open-source LLMs, models can deceive using misleading non-falsehoods, and truth
probes trained on true/false datasets detect lies much better than non-lying deception. Relevant as
evidence that truth probes key on falsity rather than deceptive intent.

**`luikham2026asymmetries`** (arXiv 2609.00180) -- Compares instructed and spontaneous deception in
Llama-3.1-70B-Instruct via direction geometry, cross-setting classifiers and steering. The two share
a direction component (cosine about 0.5) with asymmetric transfer: classifiers trained on
spontaneous deception transfer better to instructed data than the reverse.

**`moustafa2026beyond`** (arXiv 2607.20479) -- Studies how representation depth, probe expressivity,
SAE features and lie typology of training data affect deception probes, adding a dataset with
fabrication, omission and exaggeration. Optimal depth is dataset-dependent, expressive probes give
selective gains, SAE features match dense states, and training lie typology substantially changes
detectability.

**`cooney2026didyoulie`** (arXiv 2606.12618) -- Argues lie-detector evaluation needs testbeds where
models verifiably believe the opposite of what they say, and that existing model organisms often
fail this. Builds 13 reasoning model organisms with CoT-verified beliefs plus a prompted-lying
testbed, and evaluates a CoT judge, a logprob classifier and two activation probes across 31 models.
Detectors scale with capability on prompted lying but activation and logprob detectors drop sharply
on the trained organisms.

**`hopkins2026finetuned`** (Anthropic Alignment Science Blog, "Fine-Tuned Lie Detectors Failed to
Generalize", Jack Hopkins, Dipika Khullar, Rowan Wang, Fabien Roger, August 21, 2026) -- Elicits
on-policy lies in 12 settings (about 200,000 labelled examples, eight model families) using neutral
belief elicitation followed by pressure, then fine-tunes models to answer "did you lie?".
In-distribution detection works (AUROC 0.60 to 0.95) but cross-category transfer does not (about
0.70-0.75), and zero-shot larger models often beat the fine-tuned detectors. Notes that labels are
hard to trust and that some labelled lies may be reflexive or reflect confusion rather than
knowing deception.

**`parrack2025benchmarking`** (arXiv 2507.12691) -- Benchmarks deception probes by how much a
white-box monitor (with token-level probe activations) outperforms a black-box monitor. Finds weak
but encouraging black-to-white boosts from existing probes.

**`boxo2025caught`** (arXiv 2508.19505) -- Linear probes on Llama and Qwen models (1.5B-14B,
including DeepSeek-R1 finetunes) distinguish deceptive from non-deceptive arguments about factual
questions, exceeding 90% accuracy at best. Probes on 1.5B models are at chance, larger models reach
70-80%, reasoning variants above 90%.

**`campbell2023localizing`** (arXiv 2311.15131) -- Opens by noting it is often unclear whether false
outputs come from lack of knowledge or dishonesty. Studies instructed lying in LLaMA-2-70b-chat
with probing and activation patching, localising five layers and 46 attention heads whose
intervention restores honest answers.

**`pacchiardi2023catch`** (arXiv 2309.15840) -- Defines lying as outputting false statements despite
demonstrably knowing the truth. A black-box detector asks unrelated follow-up questions and feeds
yes/no answers to logistic regression; trained on GPT-3.5 instructed lies, it generalises to other
architectures, fine-tuned liars, sycophantic lies and real-life scenarios.

### Truth / belief probes

**`azaria2023internal`** (arXiv 2304.13734; Findings of EMNLP 2023) -- Trains a classifier on hidden
activations to predict whether a statement the model reads or generates is true, reaching 71-83%
accuracy depending on the model. Despite the title, it is a truthfulness probe on true/false
statements rather than a test of knowing deception.

**`marks2023geometry`** (arXiv 2310.06824) -- Using curated true/false datasets, shows via
visualisation, transfer and causal intervention that sufficiently large LLMs linearly represent
truth value. Simple difference-in-mean probes generalise across datasets.

**`burns2022discovering`** (arXiv 2212.03827) -- Contrast-Consistent Search: finds a direction in
activation space satisfying logical consistency (a statement and its negation have opposite truth
values) with no labels. Beats zero-shot accuracy by 4% on average across 6 models and 10 datasets
and remains accurate when models are prompted to output wrong answers.

**`levinstein2023still`** (arXiv 2307.00175) -- Shows the Azaria-Mitchell and CCS probes fail to
generalise in basic ways and argues on conceptual grounds that they are unlikely to work as lie
detectors even if LLMs have beliefs.

**`burger2024truth`** (arXiv 2407.12831) -- Identifies a two-dimensional subspace separating true
and false statements across Gemma-7B, LLaMA2-13B, Mistral-7B and LLaMA3-8B, explaining earlier
generalisation failures (e.g. on negations) and yielding a more robust lie detector.

**`zou2023representation`** (arXiv 2310.01405) -- Introduces representation engineering: reading and
controlling population-level representations of concepts including honesty. Source of the
contrastive honest/deceptive instruction-pair method used by later deception probes.

### Hallucination detection / knowledge vs. error

**`cheang2025really`** (arXiv 2510.09033) -- Splits hallucinations into Unassociated (no parametric
grounding) and Associated (driven by spurious learned associations). Hidden states mainly reflect
whether the model is recalling parametric knowledge, not whether the output is true, so associated
hallucinations overlap with correct answers and evade standard detectors while unassociated ones are
easily detected.

**`orgad2024llms`** (arXiv 2410.02707) -- Truthfulness information is concentrated in specific
tokens, and using them improves error detection, but such detectors do not generalise across
datasets, so truthfulness encoding is multifaceted rather than universal. Also reports a
discrepancy between internal encoding and external behaviour.

**`farquhar2024detecting`** (Nature 630:625-630, 2024) -- Semantic entropy: detects a class of
hallucinations by measuring uncertainty over meanings of sampled generations. Verified through the
Crossref record only; the abstract was not fetched, so this summary follows the description in
Kossen et al.'s abstract.

**`kuhn2023semantic`** (arXiv 2302.09664) -- Original semantic entropy method: an unsupervised
uncertainty measure accounting for semantic equivalence between generations, more predictive of QA
accuracy than baselines. Not requested, included as the methodological origin of semantic entropy.

**`kossen2024semantic`** (arXiv 2406.15927) -- Semantic entropy probes approximate semantic entropy
from the hidden states of a single generation, removing the sampling cost. They retain strong
hallucination detection and generalise out of distribution better than probes that predict accuracy
directly. The natural "hallucination probe" baseline.

**`slobodkin2023curious`** (arXiv 2310.11877; EMNLP 2023) -- When models hallucinate answers to
unanswerable questions, their hidden states still encode that the query is unanswerable, with the
first decoded token's representation a strong indicator.

**`gekhman2025insideout`** (arXiv 2503.15299) -- Defines knowledge as the fraction of
correct-incorrect answer pairs ranked correctly, computed externally (token probabilities) or
internally (intermediate computations). Models encode more knowledge internally than they express,
with an average relative gap of 40%. Useful for operationalising "the model knows".

**`simhi2024distinguishing`** (arXiv 2410.22071) -- Separates hallucinations where the model lacks
the answer (HK-) from those where it answers wrongly despite having the knowledge (HK+). HK+ cases
are prevalent, and models hallucinate on different examples, motivating model-specific detector
datasets.

**`simhi2025trust`** (arXiv 2502.12964) -- Identifies CHOKE: high-certainty hallucinations on
questions the model can otherwise answer correctly, triggered by trivial prompt perturbations.
Existing mitigations do worse on these, and a probing-based mitigation does better.

**`huang2023survey`** (arXiv 2311.05232) -- Survey of LLM hallucination: taxonomy, contributing
factors, detection methods, benchmarks and mitigation.

**`ji2022survey`** (arXiv 2202.03629) -- Earlier survey of hallucination in natural language
generation, covering metrics, mitigation and task-specific progress.

### Deception / sandbagging / sycophancy behaviour

**`vanderweij2024sandbagging`** (arXiv 2406.07358) -- Defines sandbagging as strategic
underperformance on evaluations. Frontier models can be prompted to underperform selectively on
dangerous-capability evaluations, and fine-tuned to hide capabilities behind a password.

**`sharma2023sycophancy`** (arXiv 2310.13548) -- Five AI assistants show consistent sycophancy across
four free-form tasks, and human preference data favours responses matching the user's views,
suggesting preference-based training drives it.

**`park2024deception`** (arXiv 2308.14752; Patterns 5(5):100988, 2024) -- Survey defining deception
as systematic inducement of false beliefs in pursuit of an outcome other than the truth, with
examples, risks and policy proposals.

**`scheurer2023strategically`** (arXiv 2311.07590) -- GPT-4 as a simulated trading agent acts on an
insider tip and then hides its reasons from its manager without being instructed to deceive. Source
of the insider-trading setting used by Goldowsky-Dill et al.

**`evans2021truthful`** (arXiv 2110.06674) -- Conceptual treatment of AI lying and truthfulness
standards, including a proposed standard of avoiding "negligent falsehoods" as a generalisation of
lies.

### Datasets and models

**`joshi2017triviaqa`** (arXiv 1705.03551; ACL 2017) -- TriviaQA: over 650K question-answer-evidence
triples including 95K trivia-enthusiast-authored QA pairs.

**`grattafiori2024llama3`** (arXiv 2407.21783) -- Llama 3 model family paper. Author list truncated
with "and others" in the .bib.

**`qwen2024qwen25`** (arXiv 2412.15115) -- Qwen2.5 technical report. arXiv lists the author as
"Qwen" followed by individual names; the .bib uses "Qwen Team" plus the first names and "and
others".

**`gemmateam2024gemma2`** (arXiv 2408.00118) -- Gemma 2 model paper (2B-27B parameters). arXiv lists
"Gemma Team" as first author; truncated with "and others" in the .bib.
