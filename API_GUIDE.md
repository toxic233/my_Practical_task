# UNR-Explainer API 完整使用指南

## 概述

UNR-Explainer API 是一个基于 FastAPI 构建的图神经网络可解释性服务，提供高性能的节点和链接解释功能。

### 核心特性

- 🚀 **高性能**: 异步处理架构，支持 GPU/CPU 自适应
- 💾 **智能缓存**: 模型自动缓存，避免重复加载
- 📊 **批量处理**: 支持大批量节点的并行解释
- 🔄 **实时反馈**: 流式响应，实时获取处理进度
- 🛡️ **容错机制**: 完善的错误处理和超时控制
- 📈 **统计分析**: 自动计算重要性指标的统计信息

## 安装部署

### 环境要求

- Python 3.8+
- PyTorch 1.12+
- CUDA (可选，用于 GPU 加速)

### 安装步骤

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **启动服务**
```bash
# 方式1: 直接运行
python main.py

# 方式2: 使用 uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 方式3: 生产环境
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

3. **验证安装**
```bash
curl http://localhost:8000/api/v1/health
```

## API 端点详解

### 1. 系统状态端点

#### `GET /` - 根端点
获取 API 基本信息和当前状态。

**响应示例:**
```json
{
  "message": "UNR-Explainer API服务",
  "version": "1.0.0",
  "device": "cuda:0",
  "available_models": ["Cora_graphsage_node"]
}
```

#### `GET /api/v1/health` - 健康检查
检查服务健康状态和系统资源。

**响应示例:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.123456",
  "device": "cuda:0",
  "cuda_available": true,
  "loaded_models": 3
}
```

#### `GET /api/v1/models` - 可用模型信息
获取支持的数据集、模型类型和已加载的模型。

**响应示例:**
```json
{
  "loaded_models": [
    "Cora_graphsage_node",
    "DBLP_magnn_node"
  ],
  "supported_datasets": [
    "Cora", "CiteSeer", "PubMed", "ACM", 
    "IMDB", "DBLP", "syn1", "syn2", "syn3", "syn4"
  ],
  "supported_model_types": ["graphsage", "dgi", "magnn"],
  "supported_tasks": ["node", "link"]
}
```

### 2. 解释功能端点

#### `POST /api/v1/explain/single` - 单节点解释

解释单个节点的重要性和影响子图。

**请求参数:**
| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| dataset | string | ✅ | - | 数据集名称 (大小写不敏感) |
| model | string | ✅ | - | 模型类型 (大小写不敏感) |
| task | string | ✅ | - | 任务类型 (大小写不敏感) |
| node_id | integer | ✅ | - | 目标节点ID |
| neighbors_cnt | integer | ❌ | 5 | 最近邻节点数量 |
| maxiter | integer | ❌ | 1000 | MCTS最大迭代次数 |
| c1 | float | ❌ | 1.0 | UCB1探索参数 |
| restart | float | ❌ | 0.2 | 随机重启概率 |
| perturb | float | ❌ | 0.0 | 扰动强度 |
| timeout | integer | ❌ | 60 | 超时时间(秒) |

**参数说明:**
- 所有字符串参数支持大小写不敏感输入：`dblp` = `DBLP` = `DblP`
- 数据集名称会自动标准化为正确格式
- 模型和任务名称同样支持灵活的大小写输入

**cURL 示例:**
```bash
# 标准格式
curl -X POST "http://localhost:8000/api/v1/explain/single" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "dataset=Cora&model=graphsage&task=node&node_id=0&timeout=30"

# 大小写不敏感示例
curl -X POST "http://localhost:8000/api/v1/explain/single" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "dataset=dblp&model=MAGNN&task=Node&node_id=1&timeout=30"
```

**成功响应:**
```json
{
  "node_id": 0,
  "importance": 1.2345,
  "subgraph_size": 15,
  "subgraph_nodes": [0, 1, 2, 3, 5, 8, 13, 21],
  "subgraph_edges": [
    [0, 1], [1, 2], [2, 3], [3, 5], 
    [5, 8], [8, 13], [13, 21]
  ],
  "processing_time": 5.67,
  "model_info": {
    "dataset": "Cora",
    "model_name": "graphsage",
    "task": "node",
    "num_nodes": 2708,
    "num_edges": 5429,
    "device": "cuda:0",
    "load_time": 12.34
  },
  "status": "success"
}
```

#### `POST /api/v1/explain/batch` - 批量节点解释

批量解释多个节点，支持实时进度反馈。

**请求体 (JSON):**
```json
{
  "dataset": "Cora",
  "model": "graphsage",
  "task": "node",
  "node_ids": [0, 1, 2, 3, 4],
  "neighbors_cnt": 5,
  "maxiter": 1000,
  "c1": 1.0,
  "restart": 0.2,
  "perturb": 0.0,
  "timeout": 300
}
```

**流式响应格式:**

该端点返回 Server-Sent Events (SSE) 流，包含以下类型的事件：

1. **状态更新事件:**
```json
{
  "type": "status",
  "message": "正在加载模型...",
  "progress": 0
}
```

2. **进度更新事件:**
```json
{
  "type": "progress",
  "node_id": 0,
  "result": {
    "node_id": 0,
    "importance": 1.2345,
    "subgraph_size": 15,
    "processing_time": 5.67,
    "status": "success"
  },
  "progress": 25,
  "processed": 1,
  "total": 4
}
```

3. **超时事件:**
```json
{
  "type": "timeout",
  "node_id": 2,
  "error": "处理超时(300秒)"
}
```

4. **错误事件:**
```json
{
  "type": "node_error",
  "node_id": 3,
  "error": "节点不存在"
}
```

5. **完成事件:**
```json
{
  "type": "completed",
  "final_result": {
    "request_id": "batch_1705311600000",
    "total_nodes": 5,
    "processed_nodes": 4,
    "results": [...],
    "overall_stats": {
      "importance_mean": 1.2345,
      "importance_std": 0.5678,
      "importance_max": 2.1,
      "importance_min": 0.8,
      "size_mean": 12.5,
      "size_std": 3.2,
      "size_max": 18,
      "size_min": 8,
      "avg_processing_time": 6.7,
      "success_rate": 0.8
    },
    "processing_time": 45.67,
    "status": "completed"
  },
  "progress": 100
}
```

## 支持的数据集和模型

### 数据集分类

#### 同构网络
- **Cora**: 2708个节点，5429条边，7个类别的学术论文网络
- **CiteSeer**: 3327个节点，4732条边，6个类别的学术论文网络  
- **PubMed**: 19717个节点，44338条边，3个类别的生物医学论文网络

#### 异构网络
- **ACM**: 学术会议、论文、作者的异构网络
- **IMDB**: 电影、演员、导演的异构网络
- **DBLP**: 作者、论文、期刊、会议的异构网络

#### 合成数据集
- **syn1-syn4**: 不同复杂度的合成图数据

### 模型类型

#### GraphSAGE
- **适用**: 大规模图的归纳学习
- **特点**: 邻居采样和聚合
- **支持任务**: 节点分类、链接预测

#### DGI (Deep Graph Infomax)
- **适用**: 无监督节点表示学习
- **特点**: 互信息最大化
- **支持任务**: 节点分类

#### MAGNN (Metapath Aggregated Graph Neural Network)
- **适用**: 异构图分析
- **特点**: 元路径聚合
- **支持任务**: 节点分类
- **推荐数据集**: IMDB, DBLP, ACM

### 数据集特殊处理

#### DBLP 数据集
- **节点类型**: 作者(0-4056), 论文(4057-18405), 期刊(18406-18425), 会议(18426-18447)
- **解释范围**: 仅支持作者节点解释 (node_id < 4057)
- **元路径**: APA (作者-论文-作者), APCPA (作者-论文-会议-论文-作者)

#### IMDB 数据集
- **节点类型**: 电影(0-4277), 演员(4278-16777), 导演(16778-19061)
- **解释范围**: 仅支持电影节点解释
- **元路径**: MAM (电影-演员-电影), MDM (电影-导演-电影)

## 错误处理

### HTTP 状态码

| 状态码 | 说明 | 常见原因 |
|--------|------|----------|
| 200 | 成功 | 请求正常处理 |
| 400 | 请求错误 | 参数无效、节点不存在 |
| 408 | 请求超时 | 处理时间超过设定阈值 |
| 422 | 参数验证失败 | 参数类型错误、值超出范围 |
| 500 | 服务器错误 | 模型加载失败、内部异常 |

### 错误响应格式

```json
{
  "detail": "具体错误描述信息"
}
```

### 常见错误处理

1. **节点ID无效**
```json
{
  "detail": "节点ID 9999 不存在"
}
```

2. **数据集不支持**
```json
{
  "detail": "DBLP数据集只支持作者节点(ID < 4057)"
}
```

3. **模型加载失败**
```json
{
  "detail": "模型加载失败: 文件不存在"
}
```

4. **处理超时**
```json
{
  "detail": "处理超时(60秒)"
}
```

## 性能优化

### 模型缓存策略

- **首次加载**: 模型会被完全加载并缓存
- **后续请求**: 直接使用缓存的模型，响应更快
- **内存管理**: 自动管理模型缓存，避免内存溢出

### GPU/CPU 自适应

```python
# 自动设备选择逻辑
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

- **GPU优先**: 自动检测并使用可用的 GPU
- **CPU回退**: GPU 不可用时自动使用 CPU
- **设备信息**: 通过健康检查查看当前使用设备

### 批量处理优化

- **并行处理**: 多个节点可并行解释
- **流式响应**: 实时返回结果，无需等待全部完成
- **资源控制**: 合理控制并发数量，避免资源耗尽

## 监控和日志

### 日志级别

- **INFO**: 正常操作信息
- **WARNING**: 警告信息
- **ERROR**: 错误信息和堆栈

### 关键监控指标

1. **模型加载时间**: 监控模型缓存效果
2. **请求处理时间**: 评估系统性能
3. **成功率**: 监控系统稳定性
4. **内存使用**: 防止内存泄漏
5. **GPU利用率**: 优化计算资源

### 日志示例

```
INFO:     初始化ModelManager，使用设备: cuda:0
INFO:     开始加载模型: Cora_graphsage_node
INFO:     模型加载完成: Cora_graphsage_node, 耗时: 12.34秒
ERROR:    模型加载失败: DBLP_invalid_model, 错误: 文件不存在
```

## 最佳实践

### 1. 请求优化

- **批量优于单个**: 多节点解释时使用批量接口
- **合理设置超时**: 根据数据集复杂度调整超时时间
- **参数调优**: 根据需求调整 MCTS 参数

### 2. 错误处理

- **重试机制**: 对临时性错误实施重试
- **优雅降级**: 部分失败时继续处理其他节点
- **监控告警**: 设置关键错误的告警机制

### 3. 资源管理

- **模型预热**: 预先加载常用模型组合
- **内存监控**: 定期检查内存使用情况
- **并发控制**: 避免同时处理过多请求

### 4. 数据准备

- **数据完整性**: 确保数据集文件完整可访问
- **模型匹配**: 确认模型文件与数据集匹配
- **路径配置**: 正确配置数据和模型路径

## 常见问题解答

### Q1: 如何选择合适的参数？

**A1**: 参数选择建议：

- **小图 (< 1000节点)**: maxiter=500, c1=1.0, restart=0.2
- **中等图 (1000-10000节点)**: maxiter=1000, c1=1.0, restart=0.2  
- **大图 (> 10000节点)**: maxiter=1500, c1=0.8, restart=0.3

### Q0: 遇到500错误：使用小写"dblp"等参数？

**A0**: 🔧 **已修复** - API现在支持大小写不敏感的参数：

```python
# 以下写法都是有效的
client.explain_single("dblp", "magnn", "node", 1)    # 小写
client.explain_single("DBLP", "MAGNN", "NODE", 1)    # 大写  
client.explain_single("DblP", "MaGnN", "NoDe", 1)    # 混合
```

所有参数会自动标准化为正确格式，不再需要担心大小写问题。

### Q2: 批量处理时如何处理失败的节点？

**A2**: 系统会：
- 自动跳过无效节点
- 记录失败原因
- 继续处理其他节点
- 在最终统计中排除失败节点

### Q3: 如何提高处理速度？

**A3**: 性能优化建议：
- 使用 GPU 加速
- 预加载常用模型
- 减少 maxiter 参数
- 使用批量接口

### Q4: 内存不足怎么办？

**A4**: 解决方案：
- 减少批量处理的节点数量
- 使用更小的 maxiter 值
- 清理模型缓存
- 使用 CPU 模式

### Q5: 如何验证结果的正确性？

**A5**: 验证方法：
- 检查 importance 值范围 (通常 0-5)
- 验证子图的连通性
- 对比不同参数的结果
- 使用已知结果进行验证

## API 版本信息

- **当前版本**: 1.0.0
- **FastAPI 版本**: 0.104.1
- **Python 要求**: 3.8+
- **PyTorch 要求**: 1.12+

## 联系支持

如需技术支持或报告问题，请：
1. 检查日志文件
2. 验证数据和模型文件
3. 查看本文档的故障排除部分
4. 提供详细的错误信息和环境配置 

# 运行所有示例
python example_usage.py 