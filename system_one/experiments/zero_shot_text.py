"""Zero-shot System One test of a diffusion LLM, answering in plain text.

No training and no logit reading yet: each Jev-style question is turned into
a chat prompt, the model generates a short text answer, and we parse it back
into the typed answer (bool / option key / level index). This measures how
much decision ability the backbone has out of the box.

Decoding modes (Nemotron-Labs-Diffusion is tri-mode):
  dlm_bi  diffusion decoding, prompt read bidirectionally (causal_context=False)
  dlm     diffusion decoding, prompt prefilled causally (the model's default)
  ar      ordinary autoregressive decoding

Usage:
  python zero_shot_text.py --model PATH_OR_REPO [--modes dlm_bi,dlm,ar] [--ids n1,c2]
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from zero_shot_cases import CASES  # noqa: E402

SYSTEM = (
    "You are a decision function inside a software system. Read the STATE, "
    "then answer the QUESTION using exactly one of the allowed answers. "
    "Reply with the answer only: no explanation, no punctuation."
)


def build_prompt(state, q):
    lines = [f"STATE:\n{state}\n"]
    if q["type"] == "noul":
        lines.append(f"STATEMENT: {q['instructions']}")
        lines.append("QUESTION: Is the statement true given the state?")
        lines.append("ALLOWED ANSWERS: yes, no")
    elif q["type"] == "choice":
        lines.append(f"QUESTION: {q['instructions']}")
        lines.append("OPTIONS:")
        for key, desc in q["criteria"].items():
            lines.append(f"- {key}" + (f": {desc}" if desc else ""))
        lines.append("ALLOWED ANSWERS: " + ", ".join(q["criteria"]))
    elif q["type"] == "score":
        lines.append(f"QUESTION: {q['instructions']}")
        lines.append("LEVELS:")
        for i, desc in enumerate(q["criteria"]):
            lines.append(f"{i}: {desc}")
        lines.append("ALLOWED ANSWERS: " + ", ".join(str(i) for i in range(len(q["criteria"]))))
    return "\n".join(lines)


def parse_answer(text, q):
    """Map generated text to a typed answer, or None if it doesn't parse."""
    t = text.strip().lower()
    words = re.findall(r"[a-z0-9_]+", t)
    if not words:
        return None
    if q["type"] == "noul":
        for w in words:
            if w in ("yes", "true"):
                return True
            if w in ("no", "false"):
                return False
        return None
    if q["type"] == "choice":
        keys = [k.lower() for k in q["criteria"]]
        # Prefer an exact match of the whole answer, then the first key mentioned.
        joined = "_".join(words)
        if joined in keys:
            return list(q["criteria"])[keys.index(joined)]
        for w in words:
            if w in keys:
                return list(q["criteria"])[keys.index(w)]
        return None
    if q["type"] == "score":
        for w in words:
            if w.isdigit() and int(w) < len(q["criteria"]):
                return int(w)
        return None


def generate(model, tok, prompt_ids, mode, max_new_tokens):
    if mode == "ar":
        return model.ar_generate(prompt_ids, max_new_tokens=max_new_tokens, eos_token_id=tok.eos_token_id)
    return model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        block_length=max_new_tokens,
        threshold=0.9,
        causal_context=(mode == "dlm"),
        eos_token_id=tok.eos_token_id,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--modes", default="dlm_bi,dlm,ar")
    ap.add_argument("--ids", default=None, help="comma-separated case ids to run")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=None, help="write per-case results as JSONL")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=torch.bfloat16).to(device).eval()
    print(f"loaded {args.model} on {device} in {time.time() - t0:.1f}s", flush=True)

    cases = CASES
    if args.ids:
        wanted = set(args.ids.split(","))
        cases = [c for c in CASES if c["id"] in wanted]
    modes = args.modes.split(",")

    results = []
    for case in cases:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(case["state"], case["q"])},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        for mode in modes:
            t = time.time()
            out_ids, nfe = generate(model, tok, prompt_ids, mode, args.max_new_tokens)
            dt = time.time() - t
            text = tok.decode(out_ids[0, prompt_ids.shape[1]:], skip_special_tokens=True)
            ans = parse_answer(text, case["q"])
            ok = ans == case["gold"]
            r = dict(id=case["id"], type=case["q"]["type"], tag=case["tag"], mode=mode,
                     gold=case["gold"], answer=ans, ok=ok, parsed=ans is not None,
                     text=text, nfe=nfe, seconds=round(dt, 3), prompt_tokens=prompt_ids.shape[1])
            results.append(r)
            flag = "OK  " if ok else ("FAIL" if ans is not None else "PARSE")
            print(f"[{flag}] {case['id']:<4} {mode:<6} gold={case['gold']!s:<10} got={ans!s:<10} "
                  f"nfe={nfe:<3} {dt:5.2f}s  text={text.strip()[:60]!r}", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    # Summary: accuracy per mode, by question type and by tag.
    print("\n=== summary ===")
    for mode in modes:
        rs = [r for r in results if r["mode"] == mode]
        acc = sum(r["ok"] for r in rs) / len(rs)
        parse = sum(r["parsed"] for r in rs) / len(rs)
        lat = sum(r["seconds"] for r in rs) / len(rs)
        nfe = sum(r["nfe"] for r in rs) / len(rs)
        print(f"{mode:<6} acc={acc:.2%} parsed={parse:.0%} mean_latency={lat:.2f}s mean_nfe={nfe:.1f}")
        for key in ("type", "tag"):
            groups = defaultdict(list)
            for r in rs:
                groups[r[key]].append(r["ok"])
            print("   " + "  ".join(f"{g}={sum(v)}/{len(v)}" for g, v in sorted(groups.items())))


if __name__ == "__main__":
    main()
