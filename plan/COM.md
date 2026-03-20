# 命令化一切 (COMMAND-IFY EVERYTHING)

设置并管理桥梁裂纹智能检测项目的本地运行环境。

## 1. 安装依赖

```bash
cd VBI-CrackNet && pip install -r requirements.txt
```

安装所有Python依赖包，包括NumPy、SciPy、TensorFlow/PyTorch、Matplotlib、PyYAML等核心组件。

## 2. 配置仿真参数

```bash
cp config/simulation.example.yaml config/simulation.yaml
```

编辑 `simulation.yaml` 自定义仿真参数：
- 车辆参数（质量、刚度、阻尼、速度）
- 桥梁参数（长度、弹性模量、惯性矩、阻尼比）
- 裂纹参数（位置、深度）
- 路面粗糙度等级（A~H）
- 仿真时长、时间步长

## 3. 运行仿真生成数据

```bash
python scripts/generate_data.py --config config/simulation.yaml --output data/simulation.h5
```

执行车桥耦合仿真，生成健康与损伤状态下的接触点位移响应，计算CPDV指标：
- 自动创建 `data/` 目录
- 支持批量生成不同裂纹工况的数据集
- 输出HDF5格式文件，包含CPDV序列及对应标签（位置、深度）

## 4. 训练神经网络模型

```bash
python scripts/train_model.py --data data/simulation.h5 --model models/cracknet.h5 --config config/training.yaml
```

使用BP神经网络训练损伤识别模型：
- 数据预处理（归一化、PCA降维）
- 网络结构：输入层、隐藏层、输出层
- 优化算法：Adam、早停、L2正则化
- 模型保存为HDF5格式

## 5. 评估模型性能

```bash
python scripts/evaluate_model.py --model models/cracknet.h5 --data data/simulation.h5 --output results/evaluation.json
```

在测试集上评估模型，计算MAE、RMSE、R²等指标：
- 生成预测结果对比图（位置、深度）
- 输出JSON格式的性能报告
- 可选：绘制CPDV曲线与损伤参数关系图

## 6. 验证运行状态

```bash
# 查看最新生成的仿真数据
h5ls data/simulation.h5

# 查看训练日志
tail -f logs/train.log

# 检查模型结构
python -c "from tensorflow.keras.models import load_model; model=load_model('models/cracknet.h5'); model.summary()"
```

## 核心访问点

- **仿真数据**：`data/simulation.h5`
- **模型文件**：`models/cracknet.h5`
- **配置文件**：`config/simulation.yaml`, `config/training.yaml`
- **评估结果**：`results/evaluation.json`
- **日志文件**：`logs/train.log`

## 服务管理

停止运行：
- 仿真进程：`Ctrl+C` 终止命令
- 清除临时文件：`rm -rf temp/` 或 `rm -rf data/__pycache__/`

## 注意事项

- 首次运行自动创建 `data/`、`models/`、`results/`、`logs/` 目录
- 支持GPU加速：若使用TensorFlow，确保CUDA可用
- 通过 `config/` 目录下的YAML文件自定义所有仿真与训练参数
- 模型训练时间取决于数据量和网络复杂度，建议使用GPU
- 结果可视化需安装Matplotlib，若在无GUI环境运行，设置 `matplotlib.use('Agg')`