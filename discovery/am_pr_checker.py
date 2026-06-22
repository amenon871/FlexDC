import numpy as np
import pandas as pd

p_values = np.linspace(0.2, 0.84, 7)
r_values = np.linspace(0.01, 0.84, 7)

PMAX = 763/1000
PMIN = 198/1000

ALPHA = 1.3

p_range = [0.9*PMIN , 1.1*PMAX]

allowed_values = {}

p_values = p_values[(p_values >= p_range[0]) & (p_values <= p_range[1])]

for p in p_values:
    allowed_r = []
    for r in r_values:
        baseline = min(p, ALPHA*PMAX - p)
        if r <= baseline:
            allowed_r.append(r)
    allowed_values[p] = allowed_r

print("Allowed values of P and R:")
for p, r_list in allowed_values.items():
    print(f"P = {p}: R = {r_list}")
