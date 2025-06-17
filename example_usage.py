#!/usr/bin/env python3
"""
UNR-Explainer API 使用示例

本文件展示了如何使用 UNR-Explainer API 进行图神经网络解释任务。
运行前请确保 API 服务已启动：python main.py
"""

import requests
import json
import time
from typing import List, Dict, Any


class APIClient:
    """API 客户端类"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self):
        """健康检查"""
        response = requests.get(f"{self.base_url}/api/v1/health")
        return response.json()
    
    def explain_single(self, dataset: str, model: str, task: str, node_id: int, **kwargs):
        """单节点解释"""
        params = {"dataset": dataset, "model": model, "task": task, "node_id": node_id, **kwargs}
        response = requests.post(f"{self.base_url}/api/v1/explain/single", params=params)
        response.raise_for_status()
        return response.json()
    
    def explain_batch(self, dataset: str, model: str, task: str, node_ids: List[int], **kwargs):
        """批量解释"""
        payload = {"dataset": dataset, "model": model, "task": task, "node_ids": node_ids, **kwargs}
        response = requests.post(f"{self.base_url}/api/v1/explain/batch", json=payload, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    yield json.loads(line[6:])


def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    client = APIClient()
    
    # 1. 健康检查
    health = client.health_check()
    print(f"API状态: {health['status']}")
    print(f"使用设备: {health['device']}")
    
    # 2. 单节点解释
    result = client.explain_single(
        dataset="Cora",
        model="graphsage", 
        task="node",
        node_id=0
    )
    
    print(f"节点 {result['node_id']} 解释结果:")
    print(f"  重要性: {result['importance']:.4f}")
    print(f"  子图大小: {result['subgraph_size']}")
    print(f"  处理时间: {result['processing_time']:.2f}秒")


def example_batch_processing():
    """批量处理示例"""
    print("\n=== 批量处理示例 ===")
    
    client = APIClient()
    node_ids = [0, 1, 2, 3, 4]
    
    print(f"批量解释节点: {node_ids}")
    
    for event in client.explain_batch("Cora", "graphsage", "node", node_ids):
        event_type = event.get('type')
        
        if event_type == 'status':
            print(f"状态: {event['message']}")
        elif event_type == 'progress':
            result = event['result']
            print(f"节点 {result['node_id']}: 重要性={result['importance']:.4f}")
        elif event_type == 'completed':
            stats = event['final_result']['overall_stats']
            print(f"完成! 平均重要性: {stats.get('importance_mean', 0):.4f}")


def example_parameter_tuning():
    """参数调优示例"""
    print("\n=== 参数调优示例 ===")
    
    client = APIClient()
    
    # 测试不同参数
    configs = [
        {"name": "默认参数", "params": {}},
        {"name": "快速模式", "params": {"maxiter": 500}},
        {"name": "精确模式", "params": {"maxiter": 2000, "c1": 0.8}}
    ]
    
    for config in configs:
        print(f"\n测试 {config['name']}:")
        try:
            result = client.explain_single(
                dataset="Cora", model="graphsage", task="node", node_id=0,
                **config['params']
            )
            print(f"  重要性: {result['importance']:.4f}")
            print(f"  处理时间: {result['processing_time']:.2f}秒")
        except Exception as e:
            print(f"  失败: {e}")


def example_error_handling():
    """错误处理示例"""
    print("\n=== 错误处理示例 ===")
    
    client = APIClient()
    
    error_cases = [
        {"name": "无效节点", "params": {"dataset": "Cora", "model": "graphsage", "task": "node", "node_id": 999999}},
        {"name": "DBLP非作者节点", "params": {"dataset": "DBLP", "model": "magnn", "task": "node", "node_id": 5000}},
        {"name": "大小写测试(应该成功)", "params": {"dataset": "dblp", "model": "MAGNN", "task": "Node", "node_id": 1}}
    ]
    
    for case in error_cases:
        print(f"\n测试 {case['name']}:")
        try:
            result = client.explain_single(**case['params'])
            print(f"  意外成功: {result.get('status')}")
        except requests.exceptions.HTTPError as e:
            error_detail = e.response.json().get('detail', '未知错误')
            print(f"  预期错误 ({e.response.status_code}): {error_detail}")


def example_different_datasets():
    """不同数据集示例"""
    print("\n=== 不同数据集示例 ===")
    
    client = APIClient()
    
    test_cases = [
        {"name": "Cora", "dataset": "Cora", "model": "graphsage", "task": "node", "node_id": 0},
        {"name": "CiteSeer", "dataset": "CiteSeer", "model": "graphsage", "task": "node", "node_id": 0},
        {"name": "DBLP作者", "dataset": "DBLP", "model": "magnn", "task": "node", "node_id": 100}
    ]
    
    for case in test_cases:
        print(f"\n测试 {case['name']}:")
        try:
            result = client.explain_single(**{k: v for k, v in case.items() if k != 'name'})
            print(f"  重要性: {result['importance']:.4f}")
            print(f"  模型信息: {result['model_info']['num_nodes']} 节点")
        except Exception as e:
            print(f"  失败: {e}")


if __name__ == "__main__":
    print("🚀 UNR-Explainer API 使用示例")
    print("请确保 API 服务已启动: python main.py\n")
    
    try:
        example_basic_usage()
        example_batch_processing() 
        example_parameter_tuning()
        example_error_handling()
        example_different_datasets()
        
        print("\n✅ 所有示例运行完成!")
        
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n运行错误: {e}") 