"""Zero-shot System One test of a diffusion LLM, answering in plain text.

No training and no logit reading yet: each Jev-style question is turned into
a chat prompt, the model generates a short text answer, and we parse it back
into the typed answer (bool / option key / level index). This measures how
much decision ability the backbone has out of the box.

Decoding modes (Nemotron-Labs-Diffusion is tri-mode; plain LLMs use `ar`):
  dlm_bi  diffusion decoding, prompt read bidirectionally (causal_context=False)
  dlm     diffusion decoding, prompt prefilled causally (the model's default)
  ar      ordinary autoregressive decoding

Prompt repetition (--repeats 1,2): with 2 the user message is pasted twice.
In a causal model the second copy can attend to the whole first copy, a cheap
stand-in for bidirectional reading (Leviathan et al., "Prompt Repetition
Improves Non-Reasoning LLMs").

Usage:
  python zero_shot_text.py --model PATH --backend nld --modes dlm,ar --repeats 1,2
  python zero_shot_text.py --model PATH --backend hf --repeats 1,2
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModel, AutoModelForImageTextToText, AutoTokenizer

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


def load_model(path, backend, device):
    if backend == "nld":
        return AutoModel.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16).to(device).eval()
    kwargs = {}
    config = json.load(open(Path(path) / "config.json")) if Path(path).is_dir() else {}
    if config.get("quantization_config", {}).get("quant_method") == "fp8" and device == "cpu":
        from transformers import FineGrainedFP8Config
        kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
    return AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, **kwargs).to(device).eval()


def generate(model, tok, prompt_ids, mode, max_new_tokens, backend):
    if backend == "hf":
        out = model.generate(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                             max_new_tokens=max_new_tokens, max_length=None, do_sample=False)
        return out, out.shape[1] - prompt_ids.shape[1]
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
    ap.add_argument("--backend", choices=["nld", "hf"], default="nld")
    ap.add_argument("--modes", default="ar")
    ap.add_argument("--repeats", default="1", help="comma-separated prompt repeat counts, e.g. 1,2")
    ap.add_argument("--ids", default=None, help="comma-separated case ids to run")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=None, help="write per-case results as JSONL")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    tok_kwargs = {"fix_mistral_regex": True} if "mistral" in args.model.lower() else {}
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, **tok_kwargs)
    model = load_model(args.model, args.backend, device)
    print(f"loaded {args.model} on {device} in {time.time() - t0:.1f}s", flush=True)

    cases = CASES
    if args.ids:
        wanted = set(args.ids.split(","))
        cases = [c for c in CASES if c["id"] in wanted]
    variants = [(m, int(r)) for m in args.modes.split(",") for r in args.repeats.split(",")]

    results = []
    for case in cases:
        user = build_prompt(case["state"], case["q"])
        for mode, repeat in variants:
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "\n\n".join([user] * repeat)},
            ]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
            prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            t = time.time()
            out_ids, nfe = generate(model, tok, prompt_ids, mode, args.max_new_tokens, args.backend)
            dt = time.time() - t
            text = tok.decode(out_ids[0, prompt_ids.shape[1]:], skip_special_tokens=True)
            ans = parse_answer(text, case["q"])
            ok = ans == case["gold"]
            name = f"{mode}x{repeat}"
            r = dict(id=case["id"], type=case["q"]["type"], tag=case["tag"], mode=mode, repeat=repeat,
                     variant=name, gold=case["gold"], answer=ans, ok=ok, parsed=ans is not None,
                     text=text, nfe=nfe, seconds=round(dt, 3), prompt_tokens=prompt_ids.shape[1])
            results.append(r)
            flag = "OK  " if ok else ("FAIL" if ans is not None else "PARSE")
            print(f"[{flag}] {case['id']:<4} {name:<8} gold={case['gold']!s:<10} got={ans!s:<10} "
                  f"nfe={nfe:<3} {dt:5.2f}s  text={text.strip()[:60]!r}", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    # Summary: accuracy per mode, by question type and by tag.
    print("\n=== summary ===")
    for name in dict.fromkeys(r["variant"] for r in results):
        rs = [r for r in results if r["variant"] == name]
        acc = sum(r["ok"] for r in rs) / len(rs)
        parse = sum(r["parsed"] for r in rs) / len(rs)
        lat = sum(r["seconds"] for r in rs) / len(rs)
        toks = sum(r["prompt_tokens"] for r in rs) / len(rs)
        print(f"{name:<8} acc={acc:.2%} parsed={parse:.0%} mean_latency={lat:.2f}s mean_prompt_tokens={toks:.0f}")
        for key in ("type", "tag"):
            groups = defaultdict(list)
            for r in rs:
                groups[r[key]].append(r["ok"])
            print("   " + "  ".join(f"{g}={sum(v)}/{len(v)}" for g, v in sorted(groups.items())))


if __name__ == "__main__":
    main()
