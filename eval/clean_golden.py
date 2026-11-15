# eval/clean_golden.py
import json
from collections import Counter

seen, out, first_snip = set(), [], {}
for n, line in enumerate(open("eval/golden.jsonl", encoding="utf-8"), 1):
    if not line.strip():
        continue
    r = json.loads(line)
    q, snip = r["question"].strip().lower(), r.get("expected_snippet")
    if (q, snip) in seen:
        continue                                   # exact duplicate
    if q in first_snip and first_snip[q] != snip:
        print(f"CONFLICT line {n}: {r['question']!r} has two different snippets "
              f"({first_snip[q]!r} vs {snip!r}); keeping the first")
        continue
    seen.add((q, snip)); first_snip.setdefault(q, snip); out.append(r)

with open("eval/golden.clean.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(len(out), "rows")
for (src, unans), c in sorted(Counter((r["source"], r["type"] == "unanswerable") for r in out).items(), key=str):
    print(f"  {src}{' (unanswerable)' if unans else ''}: {c}")