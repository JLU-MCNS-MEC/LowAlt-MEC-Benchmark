# 文档目录

本目录包含项目的所有详细文档，按类别组织。

## 目录结构

```
docs/
├── README.md                    # 本文档（文档索引）
├── ENVIRONMENT_SUMMARY.md       # 环境总结
├── PLOT_SCENARIO_REWARD_USAGE.md  # 绘图脚本使用说明
│
├── analysis/                    # 问题分析文档
│   ├── CLOSE_TARGET_ISSUES_ANALYSIS.md
│   ├── INITIAL_DIRECTION_ISSUE.md
│   ├── OBSERVATION_ANALYSIS.md
│   ├── PERFORMANCE_ANALYSIS.md
│   ├── POSITION_STRATEGY_ANALYSIS.md
│   ├── REWARD_ACCUMULATION_ISSUE.md
│   ├── REWARD_SCALE_ISSUE.md
│   ├── SCENARIO_20_DETAILED_ANALYSIS.md
│   ├── SCENARIO_20_FAILURE_ANALYSIS.md
│   ├── SCENARIO_SWITCH_ISSUE.md
│   ├── STRAIGHT_PATH_AND_EDGE_ISSUES.md
│   ├── SUCCESS_RATE_ANALYSIS.md
│   └── TRAINING_FAILURE_ANALYSIS.md
│
├── fixes/                       # 修复方案文档
│   ├── CLOSE_TARGET_FIXES.md
│   ├── FAST_ADAPTATION_FIXES.md
│   ├── FIXES_APPLIED.md
│   ├── REWARD_ACCUMULATION_FIX.md
│   ├── REWARD_SCALE_FIX.md
│   ├── SCENARIO_20_FIXES.md
│   ├── SCENARIO_20_FIXES_V2.md
│   └── STRAIGHT_PATH_AND_EDGE_FIXES.md
│
└── improvements/                # 训练改进文档
    ├── CURRICULUM_LEARNING_UPDATE.md
    ├── IMPROVEMENTS_APPLIED.md
    ├── REFACTORING_SUMMARY.md
    ├── SCENARIO_TRACKING_AND_SAVING.md
    ├── SCENARIO_TRACKING_UPDATE.md
    ├── TRAINING_IMPROVEMENTS.md
    └── TRAINING_IMPROVEMENTS_V2.md
```

## 文档分类说明

### 问题分析文档 (analysis/)

包含对各种训练问题的详细分析：
- **训练结果分析**: `TRAINING_RESULT_ANALYSIS.md`, `TRAINING_FAILURE_ANALYSIS.md`
- **奖励问题**: `REWARD_ACCUMULATION_ISSUE.md`, `REWARD_SCALE_ISSUE.md`
- **场景问题**: `SCENARIO_20_FAILURE_ANALYSIS.md`, `SCENARIO_SWITCH_ISSUE.md`
- **路径问题**: `STRAIGHT_PATH_AND_EDGE_ISSUES.md`, `INITIAL_DIRECTION_ISSUE.md`
- **观察空间**: `OBSERVATION_ANALYSIS.md`
- **性能分析**: `PERFORMANCE_ANALYSIS.md`, `SUCCESS_RATE_ANALYSIS.md`

### 修复方案文档 (fixes/)

包含针对各种问题的修复方案和实施细节：
- **奖励修复**: `REWARD_ACCUMULATION_FIX.md`, `REWARD_SCALE_FIX.md`
- **场景修复**: `SCENARIO_20_FIXES.md`, `SCENARIO_20_FIXES_V2.md`
- **路径修复**: `STRAIGHT_PATH_AND_EDGE_FIXES.md`, `CLOSE_TARGET_FIXES.md`
- **适配修复**: `FAST_ADAPTATION_FIXES.md`
- **修复总结**: `FIXES_APPLIED.md`

### 训练改进文档 (improvements/)

包含训练过程的改进和优化：
- **课程学习**: `CURRICULUM_LEARNING_UPDATE.md`
- **场景跟踪**: `SCENARIO_TRACKING_UPDATE.md`, `SCENARIO_TRACKING_AND_SAVING.md`
- **训练改进**: `TRAINING_IMPROVEMENTS.md`, `TRAINING_IMPROVEMENTS_V2.md`
- **重构总结**: `REFACTORING_SUMMARY.md`
- **改进总结**: `IMPROVEMENTS_APPLIED.md`

## 快速查找

### 按问题类型查找

- **奖励相关问题**: 
  - 分析: `analysis/REWARD_ACCUMULATION_ISSUE.md`, `analysis/REWARD_SCALE_ISSUE.md`
  - 修复: `fixes/REWARD_ACCUMULATION_FIX.md`, `fixes/REWARD_SCALE_FIX.md`

- **场景相关问题**:
  - 分析: `analysis/SCENARIO_20_FAILURE_ANALYSIS.md`, `analysis/SCENARIO_SWITCH_ISSUE.md`
  - 修复: `fixes/SCENARIO_20_FIXES.md`, `fixes/SCENARIO_20_FIXES_V2.md`

- **路径规划问题**:
  - 分析: `analysis/STRAIGHT_PATH_AND_EDGE_ISSUES.md`, `analysis/INITIAL_DIRECTION_ISSUE.md`
  - 修复: `fixes/STRAIGHT_PATH_AND_EDGE_FIXES.md`, `fixes/CLOSE_TARGET_FIXES.md`

### 按时间顺序查找

1. **初始训练问题**: `analysis/TRAINING_RESULT_ANALYSIS.md`
2. **奖励累积问题**: `analysis/REWARD_ACCUMULATION_ISSUE.md` → `fixes/REWARD_ACCUMULATION_FIX.md`
3. **奖励尺度问题**: `analysis/REWARD_SCALE_ISSUE.md` → `fixes/REWARD_SCALE_FIX.md`
4. **场景切换问题**: `analysis/SCENARIO_SWITCH_ISSUE.md` → `improvements/SCENARIO_TRACKING_UPDATE.md`
5. **场景20失败**: `analysis/SCENARIO_20_FAILURE_ANALYSIS.md` → `fixes/SCENARIO_20_FIXES_V2.md`

## 相关文档

- 项目根目录的 `README.md`: 项目概述和快速开始
- 项目根目录的 `DOCUMENTATION.md`: 文档索引（已迁移到本目录）
- 项目根目录的 `PROJECT_SUMMARY.md`: 项目总结

