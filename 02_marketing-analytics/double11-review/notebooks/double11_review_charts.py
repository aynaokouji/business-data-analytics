"""2026年双11复盘图表生成脚本。

数据来源：data/double11_2026.csv（每条数字均已核对公开来源，见 README.md 第六节）。
只画已核实的数据，不编造、不推测。

运行：~/workspace/.venvs/churn/bin/python notebooks/double11_review_charts.py
输出：outputs/*.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#2E6EA6"
LIGHT_BLUE = "#7FB0D9"
ORANGE = "#E07840"
GREEN = "#4C9A6C"
GRAY = "#8A8A8A"


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)


def hbar_chart(filename, title, labels, values, unit, colors=None, fmt="{:.1f}"):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = colors or [BLUE] * len(labels)
    y = range(len(labels))
    ax.barh(list(y), values, color=colors, height=0.62)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    for i, v in enumerate(values):
        ax.text(v * 1.01, i, f"{fmt.format(v)}{unit}", va="center", fontsize=11)
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_title(title, fontsize=13, loc="left", pad=12)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)


# 图1：全网 GMV 2025 vs 2026（亿元）+ 增速标注
fig, ax = plt.subplots(figsize=(9, 5))
years = ["2025年", "2026年"]
gmv = [16950, 17764]
bars = ax.bar(years, gmv, color=[LIGHT_BLUE, BLUE], width=0.5)
for b, v in zip(bars, gmv):
    ax.text(b.get_x() + b.get_width() / 2, v + 400, f"{v:,}亿元", ha="center", fontsize=12)
ax.set_ylim(0, 24000)
ax.set_title("全网 GMV：总量新高，增速大幅放缓", fontsize=13, loc="left", pad=12)
ax.text(
    0.5, 22500,
    "同比增速：2025年 +14.2% → 2026年 +4.8%（2024年为 +26.6%）",
    ha="center", fontsize=11, color="#333333",
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(left=False)
ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "01_total_gmv_growth.png", dpi=150)
plt.close(fig)

# 图2：2026年渠道结构（亿元）
hbar_chart(
    "02_channel_split.png",
    "2026年双11渠道结构（亿元）",
    ["综合电商（94.5%）", "即时零售（5.0%）", "社区团购（0.5%）"],
    [16790, 891, 83],
    "亿元",
    colors=[BLUE, GREEN, GRAY],
    fmt="{:,.0f}",
)

# 图3：首轮平台份额（%）
hbar_chart(
    "03_platform_share_first_round.png",
    "首轮平台份额（10月7日-10月31日）",
    ["天猫", "京东", "抖音", "拼多多", "快手"],
    [43.6, 25.3, 15.4, 8.3, 5.0],
    "%",
)

# 图4：货架 vs 直播（亿元，细分口径）
fig, ax = plt.subplots(figsize=(9, 5))
labels = ["货架电商", "直播电商"]
vals = [11093, 3325]
bars = ax.bar(labels, vals, color=[LIGHT_BLUE, ORANGE], width=0.5)
for b, v, g in zip(bars, vals, ["同比 +20.1%", "同比 +54.6%"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 300, f"{v:,}亿元", ha="center", fontsize=12)
    ax.text(b.get_x() + b.get_width() / 2, v / 2, g, ha="center", fontsize=11, color="white")
ax.set_ylim(0, 15000)
ax.set_title("货架电商 vs 直播电商（细分口径，亿元）", fontsize=13, loc="left", pad=12)
ax.text(
    0.5, 14000,
    "直播占两者合计（14,418亿元）的 23.1%",
    ha="center", fontsize=11, color="#333333",
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(left=False)
ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "04_shelf_vs_live.png", dpi=150)
plt.close(fig)

# 图5：首轮 TOP6 品类份额（%）
hbar_chart(
    "05_category_share_first_round.png",
    "首轮 TOP6 品类份额（10月7日-10月31日）",
    ["家用电器", "手机数码", "服装", "个护美妆", "鞋箱包", "食品饮料"],
    [15.7, 13.9, 13.5, 8.0, 6.2, 6.1],
    "%",
)

# 图6：AI 采用与信任（%）
hbar_chart(
    "06_ai_adoption_trust_gap.png",
    "AI 辅助购物：用的人多，信的人少",
    ["试过平台 AI 选购", "购物前咨询 AI 工具", "用 AI 辅助购物", "认为 AI 能精准匹配商品"],
    [84.6, 79.4, 66.7, 16.1],
    "%",
    colors=[BLUE, BLUE, BLUE, ORANGE],
)

print("charts done:", sorted(p.name for p in OUT.glob("*.png")))
