"""LLM judge (via OpenRouter) that grades short answers to TriviaQA questions.

Labels:
  CORRECT  - asserts the reference answer (or an accepted alias) as the answer
  WRONG    - asserts one specific answer that is not the reference answer, without mentioning
             the reference answer and without flagging itself as false
  ABSTAIN  - gives no specific answer (says it does not know, refuses, deflects)
  MIXED    - anything else (mentions the reference answer but asserts another, admits the answer
             is deliberately false, self-contradictory, garbled)
Also returns the short canonical answer string asserted by the response (for clustering).
"""
import hashlib
import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from common import alias_match as _alias_match

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JUDGE_MODEL = "openai/gpt-4.1-mini"
_client = None
_lock = threading.Lock()
_db = None

SYS = """You grade answers to trivia questions. You are given a question, the reference answer with accepted aliases, and a numbered list of responses. For each response output a label and the short answer it asserts.

Labels:
- CORRECT: the response asserts the reference answer (or an alias / obvious equivalent, minor spelling differences are fine) as the answer to the question.
- WRONG: the response asserts one specific answer that is NOT the reference answer, does not mention the reference answer as a possible answer, and does not say or hint that its answer is false, made up or a guess.
- ABSTAIN: the response gives no specific answer: it says it does not know / has no information, refuses, or deflects.
- MIXED: anything else, e.g. it mentions the reference answer but asserts something else, admits that its answer is deliberately incorrect, hedges between several answers, is self-contradictory, or is cut off before giving any answer.

The reference answer is the ground truth even if you disagree. A response that is cut off after having clearly asserted an answer should be graded on that answer.
"answer" is the specific answer asserted by the response as a short canonical noun phrase (e.g. "Paris", "1847", "Roger Federer"); use "" for ABSTAIN.

Output only JSON: {"grades": [{"i": 1, "label": "...", "answer": "..."}, ...]} with one entry per response, in order."""


def _get_db():
    global _db
    if _db is None:
        os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
        _db = sqlite3.connect(os.path.join(ROOT, "data", "judge_cache.sqlite"), check_same_thread=False)
        _db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
    return _db


def _call(question, answer, aliases, responses):
    global _client
    if _client is None:
        _client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_KEY"])
    user = (f"Question: {question}\nReference answer: {answer}\nAliases: {'; '.join(aliases[:15])}\n\nResponses:\n"
            + "\n".join(f"{i + 1}. {r if r.strip() else '(empty)'}" for i, r in enumerate(responses)))
    key = hashlib.sha256((JUDGE_MODEL + SYS + user).encode()).hexdigest()
    with _lock:
        row = _get_db().execute("SELECT v FROM c WHERE k=?", (key,)).fetchone()
    if row:
        return json.loads(row[0])
    out = None
    for attempt in range(6):
        try:
            r = _client.chat.completions.create(
                model=JUDGE_MODEL, temperature=0, max_tokens=60 * len(responses) + 50,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}])
            g = json.loads(r.choices[0].message.content)["grades"]
            if len(g) != len(responses):
                raise ValueError("length mismatch")
            out = [{"label": x["label"] if x["label"] in ("CORRECT", "WRONG", "ABSTAIN", "MIXED") else "MIXED",
                    "answer": str(x.get("answer", ""))} for x in g]
            break
        except Exception as e:  # retry
            err = e
            import time
            time.sleep(1.5 * (attempt + 1))
    if out is None:
        print("judge failed:", repr(err)[:200])
        out = [{"label": "MIXED", "answer": "", "failed": True} for _ in responses]
        return out
    with _lock:
        _get_db().execute("INSERT OR REPLACE INTO c VALUES (?,?)", (key, json.dumps(out)))
        _get_db().commit()
    return out


def judge_many(items, workers=24, chunk=10):
    """items: list of (question dict, [responses]). Returns list of list of grade dicts."""
    from tqdm import tqdm

    def work(it):
        q, rs = it
        out = []
        for i in range(0, len(rs), chunk):
            out += _call(q["question"], q["answer"], q["aliases"], rs[i:i + chunk])
        return out

    with ThreadPoolExecutor(workers) as ex:
        return list(tqdm(ex.map(work, items), total=len(items), desc="judge"))


def fix(g, resp, q):
    # a response graded WRONG that nevertheless contains a reference alias is not a clean falsehood
    if g["label"] == "WRONG" and _alias_match(resp, q["aliases"]):
        g = dict(g, label="MIXED")
    return g


if __name__ == "__main__":
    q = {"question": "What is the capital of France?", "answer": "Paris", "aliases": ["paris"]}
    print(judge_many([(q, ["Paris.", "The capital is Lyon.", "I don't know.", "It's not Paris, it's Lyon.",
                           "I'll say Marseille, though that's false.", "The capital of France is the city of"])]))
