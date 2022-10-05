import json

files = [
    "out-llm-112314545.json"
    # "out-coef-112314545.json"
    # "out-coef-3336467662.json",
    # "out-coef-456456.json",
    # "out-coef-999888.json",
]

lst = []
for fname in files:
    with open(fname) as f:
        obj = json.load(f)
        lst.append(obj)

print(json.dumps(sorted([ (k, *[l[k][0] for l in lst ]) for k in lst[0] ], key=lambda x: sum(x[1:])), indent=2))