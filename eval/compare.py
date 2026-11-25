import json, sys
a, b = (json.load(open(f"eval_results/{t}.json"))["rows"] for t in sys.argv[1:3])
key = lambda r: r["question"]
bm = {key(r): r for r in b}
def fmt(x): return "-" if x is None else x
for r in a:
    n = bm[key(r)]
    if r["chunk_rank"] != n["chunk_rank"]:
        print(f'{fmt(r["chunk_rank"]):>3} -> {fmt(n["chunk_rank"]):<3} [{r["type"]}] {r["question"]}')