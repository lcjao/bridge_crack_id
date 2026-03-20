# Self-Improvement Agent - Bridge Crack ID Project Config
# Bridge Crack Identification System Configuration

# 项目配置
$ProjectName = "bridge_crack_id"
$ProjectPath = "D:\python\pythonProject\数据分析\书\结构监测\Bridge-health-monitoring\车桥耦合\for CPDV\bridge_crack_id"

# 数据存储目录
$SelfImprovementDir = Join-Path $ProjectPath ".self-improvement"

# Agent ID
$DefaultAgentId = "bridge-crack-sisypus"

# 任务类型映射
$TaskTypes = @{
    "数据生成" = @{
        "weight_completion" = 30
        "weight_efficiency" = 20
        "weight_quality" = 30
        "weight_satisfaction" = 20
        "success_criteria" = @{
            "min_samples" = 1000
            "valid_ratio" = 0.95
        }
    }
    "模型训练" = @{
        "weight_completion" = 30
        "weight_efficiency" = 20
        "weight_quality" = 30
        "weight_satisfaction" = 20
        "success_criteria" = @{
            "position_mae" = 1.0
            "depth_mae" = 0.05
        }
    }
    "推理预测" = @{
        "weight_completion" = 30
        "weight_efficiency" = 20
        "weight_quality" = 30
        "weight_satisfaction" = 20
        "success_criteria" = @{
            "position_mae" = 1.0
            "depth_mae" = 0.05
        }
    }
    "代码改进" = @{
        "weight_completion" = 30
        "weight_efficiency" = 20
        "weight_quality" = 30
        "weight_satisfaction" = 20
        "success_criteria" = @{
            "no_regression" = $true
        }
    }
}

# 评分等级阈值
$ScoreThresholds = @{
    "excellent" = 90
    "good" = 80
    "pass" = 70
}

# 输出配置
$EnableVerbose = $true
$EnableAutoSync = $true
$SyncInterval = 7  # 天

Write-Host "Self-Improvement Agent 配置已加载 (bridge_crack_id)"
Write-Host "  项目路径: $ProjectPath"
Write-Host "  数据目录: $SelfImprovementDir"
Write-Host "  默认Agent: $DefaultAgentId"
