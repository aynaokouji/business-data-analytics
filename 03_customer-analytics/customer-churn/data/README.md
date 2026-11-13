# 数据说明

## 文件

- `supermarket_members.csv` — 真实会员数据，5000 行，本分析使用的数据
- `supermarket_members_synthetic.csv` — 早期演示用合成数据，2000 行（对照用，不再用于分析）
- `generate_synthetic_data.py` — 早期演示数据的生成脚本，固定随机种子 42，重新运行可复现同一份合成数据

## 真实数据来源与脱敏

`supermarket_members.csv` 是用户提供的真实会员数据，已脱敏：会员编号为 10001–15000 的匿名编号，姓名列已删除，不含电话、卡号等个人信息。公开仓库里只放这一份脱敏后的版本。

## 流失定义

`is_churn = 1`：流失会员（以数据提供方的口径为准）。
`recency_days` 是最近一次消费距今天数，是预测特征，不是流失定义本身。实测中流失与最近消费天数并非一一对应（流失组最近消费最短 1 天，留存组最长 364 天）。

## 字段

| 列名 | 含义 |
| --- | --- |
| member_id | 会员编号（匿名，10001–15000） |
| age | 年龄（岁） |
| gender | 性别（男 / 女） |
| region | 居住区域（东 / 南 / 西 / 北 / 中区） |
| membership_level | 会员等级（普通 / 银卡 / 金卡 / 钻石） |
| membership_months | 会员时长（月） |
| monthly_spend | 月均消费（元） |
| monthly_visits | 月均到店次数 |
| recency_days | 最近一次消费距今天数 |
| has_cobrand_card | 是否持有联名卡（0 / 1） |
| is_active_member | 是否活跃会员（0 / 1） |
| estimated_annual_spend | 预估年消费额（元）——与月均消费几乎不相关（r≈0.00），流失组与留存组均值几乎相同，未入模，建议核查口径 |
| is_churn | 是否流失（1=流失） |

## 早期合成数据（对照）

`supermarket_members_synthetic.csv` 是用业务直觉设定的概率模型生成的演示数据，**不是真实超市的经营数据**，仅用于展示分析流程。字段为 member_id、age、gender、membership_months、monthly_spend、monthly_visits、recency_days、points_balance、coupon_usage_rate、complaints、is_churn。
