# 基于反事实推理的异质图神经网络表示学习

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

> 一个结合反事实推理与异质图神经网络的创新表示学习框架，旨在提升模型的可解释性和泛化能力。

## 🚀 项目简介

异质图神经网络在处理包含多种类型节点和边的复杂网络数据方面表现出色，但存在模型黑盒化和结果不可解释的问题。本项目将反事实推理引入异质图神经网络，通过因果推断提升模型的解释性和鲁棒性。

### 核心问题
- 现有异质图神经网络缺乏可解释性
- 模型容易学习到虚假相关性，导致过拟合
- 缺乏对异质图中复杂因果关系的挖掘

### 解决方案
- 基于反事实推理的异质图表示学习框架
- 强化学习优化的反事实决策机制
- 多元扰动策略处理异质图的复杂结构

## ✨ 核心特性

### 🎯 增强可解释性
- **因果关系挖掘**: 通过反事实推理发掘异质图中的因果关系
- **解释子图生成**: 生成可解释的关键子图结构
- **虚假相关性消除**: 减少模型对虚假相关性的依赖

### 🚄 高效决策机制
- **强化学习优化**: 使用RL优化反事实决策过程
- **搜索空间缩减**: 有效处理异质图的指数级搜索空间
- **多元扰动策略**: 支持节点和边的多种操作类型

### 🔧 鲁棒性提升
- **分布外泛化**: 解决解释子图的OOD问题
- **结构保持**: 维护异质图的原有结构特征
- **随机扰动增强**: 提升模型的抗干扰能力

## 🏗️ 技术架构

### 系统架构图
[图4 研究流程模型图](image/图片6.png)

### 核心模块

#### 1. 反事实推理模块
```
目标: 通过最少扰动生成与现有预测相反的子图
方法: 多元扰动策略 + 邻居选择优化
输出: 解释子图 G_s 和反事实子图 G'
```

#### 2. 强化学习决策模块
```
状态: 目标节点类标签改变可能性
动作: {E_add, E_del, E_mod, V_add, V_del, V_mod}
奖励: 最小扰动次数 + 最大预测改变概率
```

#### 3. 结构保持模块
```
解释子图优化: 随机边节点补充
反事实子图增强: 循环扰动策略
目标: 解决尺寸和分布差异问题
```

## 📊 数据集与实验

### 数据集
| 数据集类型 | 数据集名称 | 用途 |
|------------|------------|------|
| 合成数据集 | Dataset1-3 | 与其他反事实推理方法对比 |
| 真实数据集 | DBLP, ACM, IMDB | 异质图处理能力评估 |

### 实验设置
- **文献数据异质网络示例**:
  
  [图1 文献数据异质网络](image/图片1.png)[图1 文献数据异质网络](image/图片2.png)

- **反事实推理应用场景**:
  
  [图2 反事实推理主要研究内容](image/图片3.png)

- **扰动操作示例**:
  
  [(a)节点扰动例子](image/图片4.png)[(b)边扰动例子](image/图片5.png)

### 评估指标
- **忠实度 (Fidelity)**: 解释子图重要性程度
- **鲁棒性 (Robustness)**: 反事实子图抗干扰能力  
- **最小性 (Minimality)**: 图编辑距离、解释大小等
- **预测准确率**: 基础分类性能
- **运行时间**: 算法效率评估

## 📈 实验结果

> **注意**: 此部分将根据实际实验结果进行填充

### 性能对比
```
TODO: 添加与基线方法的对比结果
- 传统异质图神经网络 (HAN, MAGNN, SeHGNN)
- 现有反事实推理方法 (RCExplainer, CF2, GOAt)
```

### 消融实验
```
TODO: 添加各模块贡献度分析
- 反事实推理模块效果
- 强化学习决策机制贡献
- 结构保持策略影响
```

### 可视化分析
```
TODO: 添加解释子图可视化结果
- 关键节点和边的识别
- 因果关系挖掘效果展示
- 不同数据集上的表现
```

## 🛠️ 快速开始

### 环境要求
```bash
Python >= 3.8
PyTorch >= 1.12.0
DGL >= 0.6.0
NumPy >= 1.19.0
FastAPI >= 0.104.1
```

### 安装
```bash
git clone https://github.com/your-username/counterfactual-heterogeneous-gnn.git
cd counterfactual-heterogeneous-gnn
pip install -r requirements.txt
```

### 启动API服务
```bash
# 开发模式
python main.py

# 或使用 uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 生产环境
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 使用示例

#### 1. Python API调用
```python
from src.model import CounterfactualHGNN
from src.data import load_heterogeneous_graph

# 加载异质图数据
graph = load_heterogeneous_graph('DBLP')

# 初始化模型
model = CounterfactualHGNN(
    node_types=graph.node_types,
    edge_types=graph.edge_types,
    hidden_dim=128
)

# 训练模型
model.fit(graph, epochs=100)

# 生成反事实解释
explanations = model.explain(target_nodes=[0, 1, 2])
```

#### 2. REST API调用
```bash
# 健康检查
curl http://localhost:8000/api/v1/health

# 单节点解释
curl -X POST "http://localhost:8000/api/v1/explain/single" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "dataset=Cora&model=graphsage&task=node&node_id=0&timeout=30"

# 查看可用模型
curl http://localhost:8000/api/v1/models
```

## 🔌 API服务使用指南

### 支持的数据集和模型

#### 数据集分类
| 类型 | 数据集 | 描述 | 节点数 |
|------|--------|------|--------|
| 同构网络 | Cora | 学术论文网络，7个类别 | 2708 |
| 同构网络 | CiteSeer | 学术论文网络，6个类别 | 3327 |
| 同构网络 | PubMed | 生物医学论文，3个类别 | 19717 |
| 异构网络 | ACM | 学术会议、论文、作者 | - |
| 异构网络 | IMDB | 电影、演员、导演 | 19061 |
| 异构网络 | DBLP | 作者、论文、期刊、会议 | 18448 |
| 合成数据 | syn1-syn4 | 不同复杂度的合成图 | - |

#### 模型类型
- **GraphSAGE**: 大规模图的归纳学习，支持节点分类和链接预测
- **DGI**: 无监督节点表示学习，基于互信息最大化
- **MAGNN**: 异构图分析，使用元路径聚合机制

### 主要API端点

#### 1. 单节点解释 `POST /api/v1/explain/single`

**参数说明:**
| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| dataset | string | ✅ | - | 数据集名称 (大小写不敏感) |
| model | string | ✅ | - | 模型类型 |
| task | string | ✅ | - | 任务类型 (node/link) |
| node_id | integer | ✅ | - | 目标节点ID |
| neighbors_cnt | integer | ❌ | 5 | 最近邻节点数量 |
| maxiter | integer | ❌ | 1000 | MCTS最大迭代次数 |
| timeout | integer | ❌ | 60 | 超时时间(秒) |

**响应示例:**
```json
{
  "node_id": 0,
  "importance": 1.2345,
  "subgraph_size": 15,
  "subgraph_nodes": [0, 1, 2, 3, 5, 8, 13, 21],
  "subgraph_edges": [[0, 1], [1, 2], [2, 3]],
  "processing_time": 5.67,
  "model_info": {
    "dataset": "Cora",
    "model_name": "graphsage",
    "task": "node",
    "device": "cuda:0"
  },
  "status": "success"
}
```

#### 2. 批量节点解释 `POST /api/v1/explain/batch`

支持Server-Sent Events (SSE)流式响应，实时返回处理进度。

**请求体示例:**
```json
{
  "dataset": "Cora",
  "model": "graphsage", 
  "task": "node",
  "node_ids": [0, 1, 2, 3, 4],
  "timeout": 300
}
```

**流式响应事件类型:**
- `status`: 状态更新
- `progress`: 单个节点结果
- `timeout`: 超时提醒
- `completed`: 最终结果统计

#### 3. 系统监控端点

```bash
# 健康检查
GET /api/v1/health

# 可用模型信息  
GET /api/v1/models

# API基本信息
GET /
```

### 错误处理

| HTTP状态码 | 说明 | 解决方案 |
|------------|------|----------|
| 400 | 请求参数错误 | 检查参数格式和范围 |
| 408 | 请求超时 | 增加timeout值或减少复杂度 |
| 422 | 参数验证失败 | 确认参数类型和必需参数 |
| 500 | 服务器内部错误 | 检查模型文件和系统资源 |

### 特殊数据集说明

#### DBLP数据集
- **节点类型**: 作者(0-4056), 论文(4057-18405), 期刊(18406-18425), 会议(18426-18447)
- **解释范围**: 仅支持作者节点解释 (node_id < 4057)

#### IMDB数据集  
- **节点类型**: 电影(0-4277), 演员(4278-16777), 导演(16778-19061)
- **解释范围**: 仅支持电影节点解释

### 性能优化建议

- **GPU加速**: 自动检测并使用可用GPU
- **模型缓存**: 首次加载后自动缓存，提升响应速度
- **批量处理**: 多节点解释时推荐使用批量接口
- **参数调优**: 根据图规模调整maxiter和timeout参数

### 📋 详细API文档

如需更详细的API使用说明，请参考：
- 📖 [完整API使用指南](API_GUIDE.md) - 包含所有端点详细说明、参数配置、错误处理等
- 🔧 最佳实践和性能优化建议
- ❓ 常见问题解答和故障排除

## 🎯 项目意义与应用

### 学术价值
- **理论贡献**: 首次将反事实推理系统性地应用于异质图神经网络
- **方法创新**: 提出多元扰动策略和强化学习优化框架
- **性能提升**: 显著改善模型的可解释性和鲁棒性

### 实际应用
- **推荐系统**: 提供可解释的推荐结果和原因分析
- **社交网络分析**: 识别影响用户行为的关键因素
- **生物信息学**: 发现蛋白质相互作用的因果机制
- **知识图谱**: 增强知识推理的可信度和透明度

### 技术影响
- 推动异质图神经网络向可解释AI方向发展
- 为复杂网络数据的因果分析提供新工具
- 促进反事实推理在图学习领域的应用

## 🤝 贡献指南

我们欢迎社区贡献！请参考以下方式参与项目：

1. **问题报告**: 在 [Issues](https://github.com/your-username/counterfactual-heterogeneous-gnn/issues) 中报告bug或提出功能请求
2. **功能开发**: Fork 项目并提交 Pull Request
3. **文档改进**: 帮助完善项目文档和示例
4. **测试用例**: 添加更多测试数据集和场景

## 📝 相关工作

### 异质图神经网络
- **HAN**: 分层注意力机制处理元路径
- **MAGNN**: 元路径实例聚合方法
- **SeHGNN**: 简化的语义融合模块

### 反事实推理
- **RCExplainer**: 强化学习边选择
- **CF2**: 平衡事实和反事实推理
- **GOAt**: 基于梯度的重要性计算

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

## 📞 联系方式

- **邮箱**: your-email@example.com
- **项目主页**: https://github.com/your-username/counterfactual-heterogeneous-gnn
- **论文**: [预印本链接]

---

如果这个项目对您有帮助，请给我们一个 ⭐ Star！ 
