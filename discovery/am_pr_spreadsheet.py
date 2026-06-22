import numpy as np

# For workload
PMAX = 732 / 1000
PMIN = 198 / 1000

ALPHA = 1.2
EPS = 1e-9

P_LOWER = 0.9 * PMIN
P_UPPER = 1.0 * PMAX
PR_UPPER = ALPHA * PMAX

p_values = np.linspace(P_LOWER, P_UPPER, 10)

R_LOWER = 0.01
R_POINTS_PER_P = 10

print(f"{p_values}\n")

allowed_values = {}

p_values = p_values[
    (p_values >= P_LOWER - EPS) &
    (p_values <= P_UPPER + EPS)
]

for p in p_values:
    max_allowed_r = min(
        p,
        PR_UPPER - p
    )

    if max_allowed_r < R_LOWER:
        allowed_r = []
    else:
        allowed_r = [
            round(float(r), 4)
            for r in np.linspace(R_LOWER, max_allowed_r, R_POINTS_PER_P)
        ]

    allowed_values[round(float(p), 4)] = allowed_r

print("Allowed values of P and R:")
for p, r_list in allowed_values.items():
    print(f"P = {p}: R = {r_list}")

WEIGHT_COUNT = 6
#UTILIZATION_VALUES = [0.5, 0.75, 0.8, 0.85, 0.9, 0.95]
UTILIZATION_VALUES = [0.85]

valid_pair_count = sum(len(r_list) for r_list in allowed_values.values())
data_rows_per_workload = valid_pair_count * WEIGHT_COUNT * len(UTILIZATION_VALUES)

txt_lines = []
txt_lines.append("Allowed values of P and R with P+R and P-R:\n")

print("\nAllowed values with P+R and P-R:")

for p, r_list in allowed_values.items():
    p_plus_r_list = [round(float(p + r), 4) for r in r_list]
    p_minus_r_list = [round(float(p - r), 4) for r in r_list]

    line = (
        f"P = {p}: "
        f"R = {r_list}, "
        f"P+R = {p_plus_r_list}, "
        f"P-R = {p_minus_r_list}"
    )

    print(line)
    txt_lines.append(line)

row_line = (
    f"\nData rows for this workload = {valid_pair_count} valid P/R pairs "
    f"* {len(UTILIZATION_VALUES)} utilization values "
    f"* {WEIGHT_COUNT} weight combinations = {data_rows_per_workload} rows"
)

print(row_line)
txt_lines.append(row_line)

with open("allowed_pr_values_for_workload.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(txt_lines))