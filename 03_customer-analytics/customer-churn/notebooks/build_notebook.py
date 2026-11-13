"""Build customer_churn_analysis.ipynb from the cell sources below, then
execute it with nbconvert. Run from notebooks/ with the project venv:
  ../../../../../../.venvs/churn/bin/python build_notebook.py
  ../../../../../../.venvs/churn/bin/python -m jupyter nbconvert --to notebook --execute --inplace customer_churn_analysis.ipynb
Data: data/supermarket_members.csv — real member data provided by the user,
de-identified (pseudonym member IDs, name column dropped). See data/README.md.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook(metadata={"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python", "version": "3.12.3"}})
cells = []

def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# 超市会员流失分析 | Customer Churn Prediction & Retention Analysis

> **数据说明：本项目使用真实会员数据**（`data/supermarket_members.csv`，n=5000，用户提供，已脱敏：会员编号为 10001–15000 的匿名编号，姓名列已删除，不含电话、卡号等个人信息）。早期演示版用的是合成数据（`data/supermarket_members_synthetic.csv`，种子 42，n=2000），已保留在 data/ 下作对照，不再用于本分析。以下结果是真实数据下的发现。

## 问题
超市会员不断流失，但运营不知道**哪些会员最可能流失、为什么流失、该先挽留谁**。本项目要回答：
1. 会员整体流失率是多少？流失会员和留存会员在行为上有什么不同？
2. 能否用可解释的模型（决策树）提前识别高流失风险会员？
3. 不同树深度下模型表现如何？哪条流失路径最值得干预？
4. 针对高风险会员，超市可以采取哪些挽留动作？

## 数据
- 来源：真实会员数据（已脱敏），见 `data/README.md`
- 样本：5000 名会员；目标变量 `is_churn`（1=流失）
- 特征：年龄、性别、居住区域、会员等级、会员时长（月）、月均消费、月均到店次数、最近消费天数、是否持有联名卡、是否活跃会员
- 未入模的列：`estimated_annual_spend`（预估年消费额）与月均消费几乎不相关（r≈0.00），流失组与留存组均值几乎相同，无预测信号，故排除，并在下文注明
""")

code("""import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK SC"  # 本机有该中文字体；换环境若缺字体，图表中文会显示为方框，可改回英文标签
plt.rcParams["axes.unicode_minus"] = False
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

df = pd.read_csv("../data/supermarket_members.csv")
print(df.shape)
print(df["is_churn"].value_counts(normalize=True).round(3))
print("缺失值：", int(df.isna().sum().sum()))
df.head()""")

md("## 1. 探索：流失会员和留存会员有什么不同？")
code("""num = ["age","membership_months","monthly_spend","monthly_visits","recency_days","has_cobrand_card","is_active_member","estimated_annual_spend"]
print(df.groupby("is_churn")[num].mean().round(1).T)
print("\\n按最近消费天数分组的流失率：")
df["recency_bin"] = pd.cut(df["recency_days"], [-1, 7, 30, 60, 90, 180, 365], labels=["≤7天","8–30天","31–60天","61–90天","91–180天",">180天"])
print(df.groupby("recency_bin", observed=True)["is_churn"].agg(["mean","count"]).round(3))
print("\\n按会员等级分组的流失率：")
print(df.groupby("membership_level", observed=True)["is_churn"].agg(["mean","count"]).round(3))
print("\\n按居住区域分组的流失率：")
print(df.groupby("region", observed=True)["is_churn"].agg(["mean","count"]).round(3))""")

code("""fig, ax = plt.subplots(figsize=(7, 4))
(df.groupby("recency_bin", observed=True)["is_churn"].mean() * 100).plot(kind="bar", ax=ax, color="#2e7d32")
ax.set_ylabel("流失率 (%)"); ax.set_xlabel("最近消费天数"); ax.set_title("最近越久没来，流失率越高")
ax.set_ylim(0, 100)
for i, v in enumerate(df.groupby("recency_bin", observed=True)["is_churn"].mean() * 100):
    ax.text(i, v + 2, f"{v:.0f}%", ha="center")
plt.tight_layout(); plt.savefig("../outputs/churn_rate_by_recency.png", dpi=150); plt.close()
print("saved outputs/churn_rate_by_recency.png")""")

md("## 2. 建模：决策树，比较不同树深度")
code("""features = ["age","membership_months","monthly_spend","monthly_visits","recency_days","has_cobrand_card","is_active_member"]
cats = ["gender","region","membership_level"]
X = pd.get_dummies(df[features + cats], columns=cats, drop_first=True)
y = df["is_churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

rows = []
for depth in [2, 3, 4, 5, 6, 8]:
    m = DecisionTreeClassifier(max_depth=depth, random_state=42, class_weight="balanced")
    m.fit(X_train, y_train)
    p = m.predict(X_test); proba = m.predict_proba(X_test)[:, 1]
    rows.append({"max_depth": depth,
                 "accuracy": round(accuracy_score(y_test, p), 3),
                 "precision": round(precision_score(y_test, p), 3),
                 "recall": round(recall_score(y_test, p), 3),
                 "f1": round(f1_score(y_test, p), 3),
                 "auc": round(roc_auc_score(y_test, proba), 3)})
results = pd.DataFrame(rows)
print(results.to_string(index=False))
best_depth = int(results.loc[results["f1"].idxmax(), "max_depth"])
print(f"\\n按 F1 选出的最优深度：max_depth = {best_depth}")""")

code("""fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(results["max_depth"], results["f1"], marker="o", label="F1")
ax.plot(results["max_depth"], results["auc"], marker="s", label="AUC")
ax.plot(results["max_depth"], results["recall"], marker="^", label="Recall")
ax.set_xlabel("树深度 max_depth"); ax.set_ylabel("分数"); ax.set_title("树越深不一定越好")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("../outputs/depth_comparison.png", dpi=150); plt.close()
print("saved outputs/depth_comparison.png")""")

md("## 3. 解释：最优决策树长什么样？哪些特征最重要？")
code("""best = DecisionTreeClassifier(max_depth=best_depth, random_state=42, class_weight="balanced")
best.fit(X_train, y_train)
print(export_text(best, feature_names=list(X.columns), max_depth=best_depth))
print("\\n测试集混淆矩阵 [[TN, FP],[FN, TP]]：")
print(confusion_matrix(y_test, best.predict(X_test)))""")

code("""imp = pd.Series(best.feature_importances_, index=X.columns).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7, 4))
imp.plot(kind="barh", ax=ax, color="#1565c0")
ax.set_xlabel("特征重要性"); ax.set_title("最重要的流失信号")
plt.tight_layout(); plt.savefig("../outputs/feature_importance.png", dpi=150); plt.close()
print(imp.sort_values(ascending=False).round(3).to_string())""")

md("## 4. RFM 分层：谁最值得挽留？")
code("""def qscore(s, ascending=True):
    q = pd.qcut(s, 5, labels=False, duplicates="drop")
    return q + 1 if ascending else 5 - q

df["R"] = qscore(df["recency_days"], ascending=False)  # 5 = 最近消费
df["F"] = qscore(df["monthly_visits"], ascending=True)
df["M"] = qscore(df["monthly_spend"], ascending=True)

def rfm_segment(r):
    R, F, M = r["R"], r["F"], r["M"]
    if R >= 4 and M >= 4 and F >= 3: return "High-value loyal"
    if R >= 4: return "Recently active"
    if R <= 2 and M >= 4 and F >= 3: return "Dormant high-value"
    if R <= 2 and M <= 2 and F <= 2: return "Churn risk"
    if R <= 2: return "Dormant average"
    if r["membership_months"] <= 12 and M <= 2: return "New / low spend"
    return "Average"

df["segment"] = df.apply(rfm_segment, axis=1)
order = ["High-value loyal", "Recently active", "Average", "New / low spend",
         "Dormant high-value", "Dormant average", "Churn risk"]
rfm = (df.groupby("segment")
         .agg(n=("is_churn", "size"), churn_rate=("is_churn", "mean"),
              recency_days=("recency_days", "mean"),
              monthly_spend=("monthly_spend", "mean"),
              monthly_visits=("monthly_visits", "mean"))
         .reindex(order))
print(rfm.round(3).to_string())
print("total n:", int(rfm["n"].sum()))""")

code("""colors = ["#1b5e20", "#66bb6a", "#9e9e9e", "#ffa726", "#fb8c00", "#ef5350", "#b71c1c"]
fig, ax = plt.subplots(figsize=(8, 4.5))
vals = (rfm["churn_rate"] * 100).reindex(order)
ax.bar(order, vals, color=colors)
ax.set_ylabel("Churn rate (%)")
ax.set_title("Churn rate by RFM segment (n=5000)")
ax.set_ylim(0, 70)
for i, (v, n) in enumerate(zip(vals, rfm["n"].reindex(order))):
    ax.text(i, v + 1.5, f"{v:.0f}%  (n={n})", ha="center", fontsize=9)
plt.xticks(rotation=18, ha="right")
plt.tight_layout(); plt.savefig("../outputs/rfm_churn_by_segment.png", dpi=150); plt.close()
print("saved outputs/rfm_churn_by_segment.png")""")

md("""## 5. 商业结论（真实数据下的发现）

- **先看“最近消费天数”**：≤7 天未到店的会员流失率只有 2.0%，31–60 天为 2.9%，61–90 天 4.8%，91–180 天 10.2%，>180 天升到 32.9%——沉默越久，挽回越难，干预窗口在 90–180 天。
- **模型表现**：按 F1 选出的最优树是深度 6（测试集 n=1250：准确率 0.780、精确率 0.472、召回率 0.836、F1 0.603、AUC 0.873；混淆矩阵 [[766, 234], [41, 209]]）。比合成数据版的最优深度 2 更深，说明真实数据里的流失模式更复杂。树的第一条分裂就是最近消费 ≤196.5 天，特征重要性：最近消费天数 0.490、月均消费 0.222、年龄 0.187，三者占了近九成。
- **流失会员画像**：平均 38.0 岁（留存组 48.8 岁）、月均消费 263 元（留存组 541 元）、最近消费 265 天（留存组 163 天）。普通等级会员流失率 27.2%，高于金卡 13.0% 和钻石 11.0%，但树里等级本身几乎不起直接作用——等级差异主要由消费和沉默时长解释，不要只按等级发券。
- **RFM 分层**：沉睡高价值 277 人（流失率 13%，月均消费 897 元）是最值得挽留的一组；沉睡三组共 1998 人（占 40%），其中流失风险 509 人流失率 58%、月均消费仅 122 元，基本已经流失；高价值忠诚 270 人流失率只有 1%，不用打扰。
- **建议动作**（按优先级）：
  1. 对 90–180 天未到店的会员定向推送“回归券 + 新品试吃”，在流失前触达；
  2. 对沉睡高价值 277 人做回归券加专属权益的定向挽留，对普通等级、低消费的年轻会员做等级权益提醒和升级激励，重点是把月均消费做起来，而不是只发通用券；
  3. 对月均到店次数下降（流失组 2.6 次 vs 留存组 3.1 次）且沉默时间拉长的会员做客服回访，先解决体验问题再谈促销。
- **下一步**：对沉睡高价值 277 人做挽留活动的 A/B 效果评估；`estimated_annual_spend`（预估年消费额）已定性为口径存疑并剔除（与月均消费 r=0.0027；年消费额 / 12 再除以月均消费，中位数 5.56 倍、最大值近 6 万倍），不再跟进。
""")

nb["cells"] = cells
nbf.write(nb, "customer_churn_analysis.ipynb")
print("wrote customer_churn_analysis.ipynb")
