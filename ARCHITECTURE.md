# GoldPulse Sentinel 技术架构说明（`goldx_alert.py`）

> 本文档基于当前主程序 `goldx_alert.py` 的实际实现

## 1. 项目目标与设计原则

`goldx_alert.py` 是单文件主程序，覆盖三类能力：

1. 实盘分钟级监控（默认运行模式）
2. 历史回测（`--backtest`）
3. 邮件通知（文本 + HTML + 内嵌图）

核心设计原则：

- **同一策略函数驱动回测和实盘**，减少“回测有效、实盘失真”。
- **可解释优先**：每次触发都输出 `reason` 和 `metrics`，可追溯打分项。
- **风控优先于抄底/摸顶**：先做硬过滤，再做评分触发。

---

## 2. 代码结构总览

虽然是单文件，但内部按职责自然分层：

1. **基础工具层**
   - 时间/参数解析：`_parse_int`、`_parse_float`、`_parse_hhmm_to_minutes`
   - 技术指标：`ema`、`macd_histogram`、`linear_slope`、`_quantile`
2. **数据接入层**
   - `fetch_twelvedata_1min_series`
   - `fetch_points_with_key_failover`（在 `main` 内部）
3. **策略层**
   - `compute_buy_signal`
   - `compute_sell_signal`
4. **运行控制层**
   - 回测引擎：`_handle_backtest`（在 `main` 内部）
   - 实盘循环：`while True` 主循环
5. **通知与输出层**
   - 邮件：`send_email`、`_format_email_body`、`_format_email_html`
   - 图表：`_plot_full_day`、`_plot_alert_snapshot`
   - 落盘：`alerts.csv`、`summary.json`、`full.png`

---

## 3. 关键数据模型

- `PricePoint(ts, price)`：一分钟行情点。
- `Signal(side, should_alert, reason, metrics)`：策略决策结果。
  - `side` 为 `BUY` 或 `SELL`
  - `should_alert` 为最终触发布尔值
  - `reason` 为可读解释
  - `metrics` 为机器可读明细（分项得分、距离、斜率等）
- `BacktestAlert(...)`：统一承载回测与实时绘图所需字段。
- `RuntimeConfig`：所有运行参数（监控、买入、卖出、邮件）。
- `MonitorWindow`：时区与监控窗口（支持跨天，比如 `09:00 -> 03:00`）。

---

## 4. 数据获取与 API Key 架构

### 4.1 TwelveData 客户端复用

通过 `_TD_CLIENTS` 缓存 `TDClient` 实例，同一 API Key 复用同一个客户端对象，避免每分钟重复初始化。

### 4.2 多 Key 轮询与失败切换

`fetch_points_with_key_failover` 的策略：

1. 从当前索引 Key 开始请求；
2. 成功后将“下次起始索引”移动到下一个 Key（均摊额度）；
3. 若遇到配额/鉴权/网络异常，自动切换下一 Key；
4. 全部失败后抛出最后异常。

这套机制在实盘和回测拉取历史数据都生效。

---

## 5. 配置加载与运行模式

### 5.1 配置来源

- `load_dotenv()` 从 `.env` 载入配置。
- `_load_runtime_config()` 解析并构建 `RuntimeConfig`。
- `_load_email_config()` 校验 `SMTP_*` 参数。

### 5.2 主程序模式（`main(argv)`）

- `--help`：显示用法
- `--test-email`：只验证邮件发送链路
- `--backtest`：执行回测并产出报告
- 默认：进入实盘轮询
- `--once`：只跑一轮（实盘巡检/调试）

---

## 6. 策略引擎：买入与卖出

## 6.1 公共框架

`compute_buy_signal` 与 `compute_sell_signal` 使用一致的结构：

1. 数据量校验（`MIN_BARS_TO_START` + 各窗口需求）
2. 计算局部极值、全局极值、会话极值、分位值、斜率
3. 先执行硬过滤（硬风控）
4. 进入提前通道评分（软决策）
5. 产出 `Signal`

## 6.2 买入策略（`BUY`）

目标：在低位区域内识别“下跌动能减弱+出现反弹确认”的时点。

主要机制：

- **低位区域定义**
  - 量化窗口：`QUANTILE_WINDOW_MIN`
  - 低位阈值：`Q_LOW`
  - 高位跳过：`Q_SKIP`
- **双锚点距离**
  - 全局低点距离：`dist_from_global_low_pct`
  - 会话低点距离：`dist_from_session_low_pct`
  - 会话低点“过老”时可切换为近窗口参考低点（guard 机制）
- **趋势过滤**
  - 买入高周期硬过滤：`BUY_HTF_SLOPE_*`
  - 软下跌态加严：`BUY_SOFT_DOWNTREND_*`
- **评分项（0~100+）**
  - 位置贴近低位（dist）
  - 停跌进度（stall）
  - 微反弹（micro）
  - 低点时效（age）
  - 短斜率确认（slope）
  - 惩罚项：新低未停、下跌短斜率惩罚
- **通道**
  - 主通道：低位提前（中分位通道已关闭）
  - 辅助通道：开盘抢底 `OPEN_SCOUT_*`

## 6.3 卖出策略（`SELL`，可选）

开启条件：`SELL_ENABLED=1`。

目标：在高位区域识别“上行动能减弱+出现回撤确认”的时点。

与买入对称的关键参数：

- 高位分位：`SELL_Q_HIGH` / `SELL_Q_SKIP_LOW`
- 高位距离：`SELL_MAX_DIST_FROM_GLOBAL_HIGH_PCT` / `SELL_MAX_DIST_FROM_SESSION_HIGH_PCT`
- 回撤确认：`SELL_PULLBACK_PCT` / `SELL_MICRO_PULLBACK_PCT`
- 趋势硬过滤：`SELL_HTF_SLOPE_*`
- 新高重置：`SELL_REARM_ON_NEW_HIGH_PCT`（抑制连续密集卖点）
- 开盘抢卖：`SELL_OPEN_SCOUT_*`

## 6.4 买卖信号选择

`_choose_signal` 逻辑：

1. 若未开卖出，只返回买入信号；
2. 若只一侧触发，返回该侧；
3. 若买卖同时触发，比较 `signal_score - score_threshold`，返回边际更强的一侧。

---

## 7. 回测引擎设计（`--backtest`）

`_handle_backtest` 的核心特性：

1. **定向历史拉取**：按 `start_date/end_date/outputsize` 请求 TwelveData。
2. **缓存机制**：
   - 读缓存：`backtest_cache/<symbol>/<date>.json`
   - 缓存覆盖不足自动重拉
   - `--refresh-cache` 强制重拉，`--no-cache` 禁用缓存
3. **无未来函数**：
   - 按时间顺序逐点推进 `seen`；
   - 每个时刻只用当前及历史窗口做判断。
4. **与实盘一致的风控约束**：
   - 冷却时间
   - 开盘通道次数上限
   - 首个买点保护
   - 会话内卖点重置条件
5. **产物**：
   - `alerts.csv`
   - `summary.json`（含完整参数快照）
   - `full.png`

---

## 8. 实盘循环设计（默认模式）

实盘主循环按以下顺序执行：

1. 读取 `STATE_FILE` 恢复上次提醒时间（冷却可跨重启）。
2. 判断是否处于监控窗口：
   - 不在窗口内：计算下个起点并休眠；
   - 在窗口内：继续采集与判定。
3. 拉取数据（多 Key 轮询/切换）。
4. 计算信号并打印日志。
5. 应用告警闸门：
   - 首个买点保护
   - 冷却与买入冷却豁免
   - 开盘通道会话上限
   - 卖出新高重置
6. 发送邮件（可内嵌本次提醒图）。
7. 更新 `STATE_FILE` 与会话状态变量。

---

## 9. 邮件与可解释性系统

## 9.1 邮件格式

- 纯文本：`_format_email_body`
- HTML 卡片：`_format_email_html`
- 支持 CID 内嵌图（不是附件下载）

邮件内容包含：

- 方向（买入/卖出）
- 推荐等级（A/B/C/D）
- 总分/阈值/分差
- 触发通道
- 分项得分明细
- 推荐动作
- 触发细则（完整 reason）

## 9.2 评分等级

`_signal_recommendation` 按 `margin = score - threshold` 分级：

- `A`: `margin >= 12`
- `B`: `margin >= 7`
- `C`: `margin >= 3`
- `D`: 其余

---

## 10. 时段与会话模型

监控时段由 `MONITOR_START` / `MONITOR_END` 决定，支持两种类型：

1. 非跨天窗口（例如 `09:00 -> 22:00`）
2. 跨天窗口（例如 `09:00 -> 03:00`）

窗口语义为左闭右开 `[start, end)`，因此如果需要覆盖“次日 02:00~02:59”，结束应设为 `03:00`。

会话锚点函数 `_session_start_anchor` 统一用于：

- 买卖策略内的“会话高/低点”计算
- 回测和实盘中的会话内计数器重置

---

## 11. 关键运行产物与目录

- 实盘日志：`logs/goldx.log`
- 实盘提醒图：`alert_out/<symbol>/<date>/alert_<side>_<YYYY-mm-dd_HHMM>.png`
- 回测缓存：`backtest_cache/<symbol>/<date>.json`
- 回测结果：`<outdir>/<symbol>/<date>/`
  - `alerts.csv`
  - `summary.json`
  - `full.png`
- 冷却状态：`.goldx_state.json`（默认）
