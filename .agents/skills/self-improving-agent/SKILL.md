# Self-Improvement Agent Skill

**Self-evaluating skill for Bridge Crack Identification System**

---

## 激活条件

当需要从任务执行中学习并自我优化时激活此技能：
- "自我改进"、"自我优化"、"学习经验"
- "任务评估"、"性能分析"
- 任务完成或失败后自动记录

---

## 项目适配配置

**项目**: Bridge Crack Identification System (bridge_crack_id)
**路径**: `D:\python\pythonProject\数据分析\书\结构监测\Bridge-health-monitoring\车桥耦合\for CPDV\bridge_crack_id`

### 评估数据存储
```
.self-improvement/
├── evaluations.json       # 任务评估记录
├── lessons-learned.json   # 经验教训库
├── optimization-plan.json  # 优化计划
└── performance-metrics.json # 性能指标
```

---

## 评分标准

### 评分公式
```
总分 = 完成度(30%) + 效率(20%) + 质量(30%) + 满意度(20%)
```

### 评分等级
- **≥ 90分**: 优秀 → 记录最佳实践
- **80-89分**: 良好 → 继续保持
- **70-79分**: 及格 → 识别改进点
- **< 70分**: 不及格 → 触发深度反思

### 科学仿真项目特定指标
- ✅ 数值稳定性 (NaN/Inf检查)
- ✅ 收敛性 (Newmark-β迭代)
- ✅ 训练质量 (MAE/RMSE达标)
- ✅ 代码效率 (仿真耗时)

---

## 使用方法

### PowerShell 脚本调用

```powershell
# 任务完成后评估
$metrics = @{
    completion = 90
    efficiency = 85
    quality = 80
    satisfaction = 85
}

# 调用评估脚本 (相对于项目目录)
$projectRoot = "D:\python\pythonProject\数据分析\书\结构监测\Bridge-health-monitoring\车桥耦合\for CPDV\bridge_crack_id"
& "$projectRoot\.agents\skills\self-improving-agent\self-improving-agent-master\scripts\evaluate-task.ps1" `
    -AgentId "bridge-crack-agent" `
    -TaskId "train-bp-$(Get-Date -Format 'yyyyMMdd-HHmmss')" `
    -TaskType "模型训练" `
    -Metrics $metrics
```

### 记录经验
```powershell
& "$projectRoot\.agents\skills\self-improving-agent\self-improving-agent-master\scripts\learn-lesson.ps1" `
    -AgentId "bridge-crack-agent" `
    -Lesson "使用np.clip防止数值溢出" `
    -Impact "high" `
    -Category "numerical-stability"
```

---

## 项目最佳实践

### 科学仿真类任务
1. **数值稳定性优先** - 始终检查NaN/Inf
2. **确定性结果** - 使用固定随机种子(42)
3. **质量指标明确** - MAE < 1.0m, 深度MAE < 0.05
4. **优雅降级** - 迭代失败时使用fallback策略

### 神经网络训练任务
1. 记录训练曲线和验证损失
2. 早停机制检查
3. 梯度裁剪防止爆炸
4. 定期保存模型checkpoint

### 数据生成任务
1. 验证健康状态分析成功
2. 检查CPDV有效性
3. 记录无效样本比例

---

## 经验分类标签

| 类别 | 示例 |
|------|------|
| `numerical-stability` | NaN/Inf处理、矩阵求逆 |
| `optimization` | 训练加速、内存优化 |
| `quality` | 精度提升、误差分析 |
| `workflow` | 流水线改进、脚本自动化 |
| `domain-knowledge` | 车桥耦合、CPDV理论 |

---

## 版本
- **Skill Version**: 1.0.0 (项目适配版)
- **项目**: bridge_crack_id v1.0
- **领域**: 科学仿真 / 神经网络

---

*让每个任务都成为学习的契机！*
