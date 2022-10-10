import json

files = [
    "out-llm-9876543.json"
]

lst = []
for fname in files:
    with open(fname) as f:
        obj = json.load(f)
        lst.append(obj)

print(json.dumps(sorted([ (k, *[l[k][0] for l in lst ]) for k in lst[0] ], key=lambda x: sum(x[1:])), indent=2))
