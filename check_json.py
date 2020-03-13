import json
import math

json_path = 'init_or_not.json'
with open(json_path, 'r') as f:
    data = json.load(f)

for key in data:
    arrs = data[key]
    avg = 0
    sd = 0

    for arr in arrs:
        sum_v = sum(arr)
        a = sum_v / len(arr)
        s = 0
        for x in arr:
            s += (a - x) ** 2
        s = math.sqrt(s)
        s /= len(arr)
        avg += a
        sd += s

    avg /= len(arrs)
    sd /= len(arrs)
    print(key, avg, sd)
