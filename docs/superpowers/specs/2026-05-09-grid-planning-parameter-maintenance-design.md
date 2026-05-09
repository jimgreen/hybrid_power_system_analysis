# 电网规划 Web 参数维护设计

日期：2026-05-09

## 背景

在现有 `power_plan` 轻量 Web 服务基础上新增“电网规划 Web 网站”的参数维护能力。网站最终包含系统首页、参数维护、启动优化、方案评估、结果比对五个主功能；本设计先聚焦参数维护模块，并为后续优化、评估、比对模块预留数据接口和方案目录结构。

## 目标

- 用原生 HTML/CSS/JavaScript + 现有 Python `http.server` 风格后端实现，不新增前端构建链。
- 每个规划方案保存为一个文件夹，文件夹名就是方案名称。
- 每个方案文件夹内保存一个 `parameters.xlsx` 工作簿，作为参数维护的唯一持久化文件。
- 参数维护界面支持显示、编辑、保存、复制、新建、重命名方案。
- 支持 8760 点风速、太阳辐射、负荷数据的显示和修改。
- 支持多类设备参数表和设计容量上下限约束的显示和修改。

## 非目标

- 本阶段不实现启动优化算法、方案评估算法和结果比对计算。
- 本阶段不接入数据库。
- 本阶段不实现多人并发编辑锁；保存时采用最后一次保存覆盖当前方案文件。
- 本阶段不实现复杂 Excel 公式、宏、图表或模板美化，只保证数据结构稳定、可读、可编辑。

## 页面与导航

新增 `power_plan/planning.html` 作为电网规划 Web 入口。页面顶部主导航包含：

1. 系统首页
2. 参数维护
3. 启动优化
4. 方案评估
5. 结果比对

当前阶段“参数维护”为可用模块，其余模块显示占位状态，后续按同一方案目录读取输入和输出。

## 参数维护界面布局

页面采用工程软件风格的三栏结构：

- 左侧：方案管理区
  - 方案列表
  - 新建方案
  - 复制方案
  - 重命名方案
  - 保存方案
  - 当前方案保存状态提示
- 中间：参数编辑区
  - 一级标签：8760时序数据、设备参数、设计容量约束汇总
  - 8760时序数据页采用上下结构：上方为 8760 点曲线板，下方为小时级数据表。
  - 设备参数页集中展示柴发、风机、光伏、储能PCS、储能电池组、电制氢、储氢罐、燃料电池，全部采用表格方式编辑。
- 右侧：校验与摘要区
  - 当前方案名称
  - 8760 行数校验
  - 各设备参数数量统计
  - 容量上下限校验结果
  - 最近保存时间

## 参数页面组织

### 8760时序数据页

采用上下结构：

- 上方为 8760 点曲线板，显示风速、太阳辐射、负荷三条全年曲线。
- 下方为小时级数据表，显示并编辑小时序号、时间、风速、太阳辐射、负荷。

### 设备参数页

柴发、风机、光伏、储能PCS、储能电池组、电制氢、储氢罐、燃料电池全部整合进同一个“设备参数”网页，采用表格方式维护。

页面内按设备类型分成多个表格区块，纵向排列或通过页内锚点快速定位：

- 柴发表格：名称、容量、设计容量下限、设计容量上限、成本、功率上限、功率下限、油耗率。
- 风机表格：名称、容量、设计容量下限、设计容量上限、成本、切入风速、切出风速。
- 光伏表格：名称、容量、设计容量下限、设计容量上限、成本、切入风速、切出风速。
- 储能PCS表格：名称、功率容量、设计容量下限、设计容量上限、成本。
- 储能电池组表格：名称、电池容量、设计容量下限、设计容量上限、成本。
- 电制氢表格：名称、功率容量、设计容量下限、设计容量上限、成本、电-氢效率。
- 储氢罐表格：名称、储氢罐容量、设计容量下限、设计容量上限、成本。
- 燃料电池表格：名称、功率容量、设计容量下限、设计容量上限、成本、氢-电效率。

每个表格均支持新增行、删除行、单元格编辑。设计容量下限和设计容量上限跟随各设备表显示和修改，不再需要跳转到独立容量约束表维护。页面顶部提供设备类型快捷入口和保存状态提示，便于在一个网页内集中维护全部设备参数。

### 设计容量约束汇总

设计容量下限和上限跟随设备参数表维护。此区域仅作为汇总视图，集中显示柴发、风机、光伏、储能PCS、储能电池组、电制氢、储氢罐、燃料电池的容量约束，并提示下限大于上限等错误。

## XLSX 存储结构

根目录：`power_plan/planning_schemes/`

每个方案目录：`power_plan/planning_schemes/<方案名称>/`

每个方案参数文件：`power_plan/planning_schemes/<方案名称>/parameters.xlsx`

工作簿包含以下工作表。

### 8760时序数据

字段：

- `hour_index`：1 到 8760
- `datetime`：可选时间字符串
- `wind_speed`：风速，单位 m/s
- `solar_irradiance`：太阳辐射，单位 W/m2
- `load`：负荷，单位 kW

校验：

- 必须正好 8760 行。
- `hour_index` 必须为 1 到 8760 且不可重复。
- 数值字段允许为空但保存时标记警告；优化计算前必须补齐。

### 柴发参数

字段：

- `name`
- `capacity`
- `design_capacity_lower`
- `design_capacity_upper`
- `cost`
- `power_upper`
- `power_lower`
- `fuel_rate`

### 风机参数

字段：

- `name`
- `capacity`
- `design_capacity_lower`
- `design_capacity_upper`
- `cost`
- `cut_in_wind_speed`
- `cut_out_wind_speed`

### 光伏参数

字段：

- `name`
- `capacity`
- `design_capacity_lower`
- `design_capacity_upper`
- `cost`
- `cut_in_wind_speed`
- `cut_out_wind_speed`

说明：用户明确要求光伏参数包含切入风速、切出风速，本设计按要求保留这两个字段，即使后续模型可能将其改名为辐照阈值或运行阈值。

### 储能PCS参数

字段：

- `name`
- `power_capacity`
- `design_capacity_lower`
- `design_capacity_upper`
- `cost`

### 储能电池组参数

字段：

- `name`
- `battery_capacity`
- `design_capacity_lower`
- `design_capacity_upper`
- `cost`

### 电制氢参数

字段：

- `name`
- `power_capacity`
- `design_capacity_lower`
- `design_capacity_upper`
- `cost`
- `electric_to_hydrogen_efficiency`

### 储氢罐参数

字段：

- `name`
- `hydrogen_tank_capacity`
- `design_capacity_lower`
- `design_capacity_upper`
- `cost`

### 燃料电池参数

字段：

- `name`
- `power_capacity`
- `design_capacity_lower`
- `design_capacity_upper`
- `cost`
- `hydrogen_to_electric_efficiency`

### 设计容量约束汇总

字段：

- `device_type`
- `design_capacity_lower`
- `design_capacity_upper`

行枚举：

- 风机容量
- 光伏容量
- 储能PCS容量
- 储能电池组容量
- 电制氢容量
- 储氢罐容量
- 燃料电池容量

校验：

- 各设备表中的设计容量下限和设计容量上限均为数值。
- 同一设备条目的设计容量下限不得大于设计容量上限。

## API 设计

所有接口路径以 `/api/planning` 开头。

- `GET /api/planning/schemes`：返回方案列表和当前可读状态。
- `POST /api/planning/schemes`：新建方案，创建文件夹和默认 `parameters.xlsx`。
- `POST /api/planning/schemes/copy`：复制方案文件夹。
- `POST /api/planning/schemes/rename`：重命名方案文件夹。
- `GET /api/planning/schemes/<name>`：读取方案参数，返回 JSON 给前端表格。
- `PUT /api/planning/schemes/<name>`：保存当前方案参数到 `parameters.xlsx`。

后端读取 XLSX 后转换为 JSON：

```json
{
  "scheme": "方案A",
  "time_series": [],
  "diesel_generators": [],
  "wind_turbines": [],
  "photovoltaics": [],
  "storage_pcs": [],
  "storage_battery_packs": [],
  "hydrogen_electrolyzers": [],
  "hydrogen_storage": [],
  "fuel_cells": [],
  "capacity_limits": [],
  "validation": []
}
```

## 依赖

新增 Python 依赖：

- `openpyxl>=3.1.0`

原因：标准库无法读写 `.xlsx`，`openpyxl` 足够轻量，适合当前无数据库、无前端构建链的方案。

## 前端交互

- 页面加载时调用方案列表接口。
- 若不存在方案，显示“新建第一个方案”。
- 选择方案后读取参数 JSON 并填充表格。
- 8760 时序数据页上方显示风速、太阳辐射、负荷三条全年曲线，支持图例开关、悬浮查看小时点数值、按小时区间缩放。
- 8760 时序数据页下方显示小时级数据表，字段包括小时序号、时间、风速、太阳辐射、负荷。
- 8760 表格支持分页或虚拟滚动，默认每页显示 168 行，支持跳转页码和按小时区间筛选；修改表格数据后，上方曲线同步刷新。
- 所有表格单元格可直接编辑。
- 设备参数页在同一个网页内用多个表格编辑柴发、风机、光伏、储能PCS、储能电池组、电制氢、储氢罐、燃料电池，支持新增行、删除行、单元格编辑；设计容量下限和上限跟随对应设备表同步显示和修改。
- 保存前执行前端基础校验；后端保存时再执行一次结构校验。
- 保存成功后更新最近保存时间和校验摘要。

## 错误处理

- 方案名为空、包含路径分隔符或非法字符时拒绝创建或重命名。
- 目标方案已存在时拒绝新建、复制或重命名。
- 读取缺失工作表时返回错误，并提示用户复制默认模板或新建方案。
- 保存失败时不删除旧文件；后端先写入临时文件，成功后替换 `parameters.xlsx`。
- XLSX 行数或字段不完整时返回校验警告，但尽量保留用户数据。

## 测试计划

- 单元测试：方案名校验、默认工作簿创建、XLSX 读写往返、容量上下限校验。
- API 测试：新建方案、复制方案、重命名方案、读取方案、保存方案。
- 手工浏览器测试：打开 `planning.html`，完成新建、编辑 8760 数据第一页、编辑设备参数、保存、刷新后数据仍存在。

## 后续扩展

- 启动优化模块读取同一方案目录下的 `parameters.xlsx`，输出优化结果到方案目录下的 `optimization_result.xlsx` 或 `results.json`。
- 方案评估模块读取优化结果并生成指标摘要。
- 结果比对模块选择多个方案目录，比较容量配置、成本、可再生能源消纳、弃风弃光、电氢转换等指标。












