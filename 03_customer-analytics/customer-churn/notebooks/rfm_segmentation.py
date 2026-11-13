"""RFM segmentation of supermarket members (real de-identified data, n=5000).

Run from notebooks/ with the project venv:
  ../../../../../../.venvs/churn/bin/python rfm_segmentation.py

Method (all thresholds documented, no hidden steps):
- R, F, M scores are quintiles (1-5) of recency_days (5 = most recent),
  monthly_visits (5 = most frequent) and monthly_spend (5 = highest spend).
  monthly_visits is discrete, so its quintiles collapse to 4 levels.
- Segments (partition: every member lands in exactly one):
    高价值忠诚 High-value loyal: R>=4, M>=4, F>=3
    近期活跃 Recently active:    R>=4, otherwise
    沉睡高价值 Dormant high-value: R<=2, M>=4, F>=3
    流失风险 Churn risk:        R<=2, M<=2, F<=2
    沉睡一般 Dormant average:   R<=2, otherwise
    新客/低消费 New / low spend: R==3, membership_months<=12, M<=2
    一般 Average:               the rest (R==3, otherwise)
- Output: prints the segment table and saves
  ../outputs/rfm_churn_by_segment.png
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("../data/supermarket_members.csv")

def qscore(s, ascending=True):
    q = pd.qcut(s, 5, labels=False, duplicates="drop")
    return q + 1 if ascending else 5 - q

df["R"] = qscore(df["recency_days"], ascending=False)
df["F"] = qscore(df["monthly_visits"], ascending=True)
df["M"] = qscore(df["monthly_spend"], ascending=True)

def segment(r):
    R, F, M = r["R"], r["F"], r["M"]
    if R >= 4 and M >= 4 and F >= 3:
        return "High-value loyal"
    if R >= 4:
        return "Recently active"
    if R <= 2 and M >= 4 and F >= 3:
        return "Dormant high-value"
    if R <= 2 and M <= 2 and F <= 2:
        return "Churn risk"
    if R <= 2:
        return "Dormant average"
    if r["membership_months"] <= 12 and M <= 2:
        return "New / low spend"
    return "Average"

df["segment"] = df.apply(segment, axis=1)
order = ["High-value loyal", "Recently active", "Average", "New / low spend",
         "Dormant high-value", "Dormant average", "Churn risk"]
table = (df.groupby("segment")
           .agg(n=("is_churn", "size"), churn_rate=("is_churn", "mean"),
                recency_days=("recency_days", "mean"),
                monthly_spend=("monthly_spend", "mean"),
                monthly_visits=("monthly_visits", "mean"))
           .reindex(order))
print(table.round(3).to_string())
print("total n:", int(table["n"].sum()),
      "weighted churn:", round((table["n"] * table["churn_rate"]).sum() / len(df), 3))
assert int(table["n"].sum()) == len(df) == 5000

colors = ["#1b5e20", "#66bb6a", "#9e9e9e", "#ffa726", "#fb8c00", "#ef5350", "#b71c1c"]
fig, ax = plt.subplots(figsize=(8, 4.5))
vals = (table["churn_rate"] * 100).reindex(order)
ax.bar(order, vals, color=colors)
ax.set_ylabel("Churn rate (%)")
ax.set_title("Churn rate by RFM segment (n=5000)")
ax.set_ylim(0, 70)
for i, (v, n) in enumerate(zip(vals, table["n"].reindex(order))):
    ax.text(i, v + 1.5, f"{v:.0f}%  (n={n})", ha="center", fontsize=9)
plt.xticks(rotation=18, ha="right")
plt.tight_layout()
plt.savefig("../outputs/rfm_churn_by_segment.png", dpi=150)
plt.close()
print("saved outputs/rfm_churn_by_segment.png")
