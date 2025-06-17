#!/usr/bin/env python3
"""
UNR-Explainer API 测试客户端
"""

import requests
import json
import time
import asyncio
import aiohttp
from typing import List, Dict, Any

class UNRExplainerClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        response = self.session.get(f"{self.base_url}/api/v1/health")
        response.raise_for_status()
        return response.json()
    
    def get_available_models(self) -> Dict[str, Any]:
        """获取可用模型"""
        response = self.session.get(f"{self.base_url}/api/v1/models")
        response.raise_for_status()
        return response.json()
    
    def explain_single_node(
        self,
        dataset: str,
        model: str,
        task: str,
        node_id: int,
        **kwargs
    ) -> Dict[str, Any]:
        """解释单个节点"""
        params = {
            "dataset": dataset,
            "model": model,
            "task": task,
            "node_id": node_id,
            **kwargs
        }
        
        response = self.session.post(
            f"{self.base_url}/api/v1/explain/single",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def explain_batch_stream(
        self,
        dataset: str,
        model: str,
        task: str,
        node_ids: List[int],
        **kwargs
    ):
        """批量解释节点（流式响应）"""
        payload = {
            "dataset": dataset,
            "model": model,
            "task": task,
            "node_ids": node_ids,
            **kwargs
        }
        
        response = self.session.post(
            f"{self.base_url}/api/v1/explain/batch",
            json=payload,
            stream=True
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = json.loads(line[6:])  # 移除 'data: ' 前缀
                    yield data

def test_basic_functionality():
    """测试基本功能"""
    client = UNRExplainerClient()
    
    print("=== 测试健康检查 ===")
    try:
        health = client.health_check()
        print(f"健康状态: {health}")
        print(f"CUDA可用: {health.get('cuda_available', False)}")
        print(f"设备: {health.get('device', 'unknown')}")
    except Exception as e:
        print(f"健康检查失败: {e}")
        return False
    
    print("\n=== 测试可用模型 ===")
    try:
        models_info = client.get_available_models()
        print(f"支持的数据集: {models_info['supported_datasets']}")
        print(f"支持的模型: {models_info['supported_model_types']}")
        print(f"支持的任务: {models_info['supported_tasks']}")
        print(f"已加载的模型: {models_info['loaded_models']}")
    except Exception as e:
        print(f"获取模型信息失败: {e}")
        return False
    
    return True

def test_single_node_explanation():
    """测试单节点解释"""
    client = UNRExplainerClient()
    
    print("\n=== 测试单节点解释 ===")
    
    # 测试参数
    test_cases = [
        {
            "dataset": "Cora", 
            "model": "graphsage", 
            "task": "node", 
            "node_id": 0,
            "timeout": 30
        },
        {
            "dataset": "DBLP", 
            "model": "magnn", 
            "task": "node", 
            "node_id": 100,  # 作者节点
            "timeout": 30
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- 测试用例 {i+1}: {test_case['dataset']} ---")
        try:
            start_time = time.time()
            result = client.explain_single_node(**test_case)
            elapsed_time = time.time() - start_time
            
            print(f"节点ID: {result['node_id']}")
            print(f"重要性: {result['importance']:.4f}")
            print(f"子图大小: {result['subgraph_size']}")
            print(f"处理时间: {result['processing_time']:.2f}秒")
            print(f"总耗时: {elapsed_time:.2f}秒")
            print(f"状态: {result['status']}")
            
        except Exception as e:
            print(f"测试失败: {e}")

def test_batch_explanation():
    """测试批量解释"""
    client = UNRExplainerClient()
    
    print("\n=== 测试批量解释 ===")
    
    # 测试参数
    test_case = {
        "dataset": "Cora",
        "model": "graphsage",
        "task": "node",
        "node_ids": [0, 1, 2, 3, 4],  # 测试5个节点
        "timeout": 60
    }
    
    print(f"开始批量解释 {len(test_case['node_ids'])} 个节点...")
    
    try:
        results = []
        for response_data in client.explain_batch_stream(**test_case):
            response_type = response_data.get('type')
            
            if response_type == 'status':
                print(f"状态: {response_data['message']} (进度: {response_data.get('progress', 0)}%)")
            
            elif response_type == 'progress':
                node_id = response_data['node_id']
                result = response_data['result']
                progress = response_data['progress']
                processed = response_data['processed']
                total = response_data['total']
                
                print(f"节点 {node_id}: 重要性={result['importance']:.4f}, "
                      f"大小={result['subgraph_size']}, "
                      f"时间={result['processing_time']:.2f}s "
                      f"({processed}/{total}, {progress}%)")
                
                results.append(result)
            
            elif response_type == 'timeout':
                print(f"节点 {response_data['node_id']} 超时: {response_data['error']}")
            
            elif response_type == 'node_error':
                print(f"节点 {response_data['node_id']} 出错: {response_data['error']}")
            
            elif response_type == 'completed':
                final_result = response_data['final_result']
                print(f"\n批量处理完成!")
                print(f"总处理节点: {final_result['total_nodes']}")
                print(f"成功处理: {final_result['processed_nodes']}")
                print(f"总耗时: {final_result['processing_time']:.2f}秒")
                
                if final_result['overall_stats']:
                    stats = final_result['overall_stats']
                    print(f"平均重要性: {stats.get('importance_mean', 0):.4f}")
                    print(f"平均子图大小: {stats.get('size_mean', 0):.2f}")
                    print(f"成功率: {stats.get('success_rate', 0)*100:.1f}%")
            
            elif response_type == 'error':
                print(f"批量处理失败: {response_data['message']}")
                break
    
    except Exception as e:
        print(f"批量测试失败: {e}")

def interactive_test():
    """交互式测试"""
    client = UNRExplainerClient()
    
    print("\n=== 交互式测试 ===")
    print("可用的数据集: Cora, CiteSeer, PubMed, ACM, IMDB, DBLP, syn1-syn4")
    print("可用的模型: graphsage, dgi, magnn")
    print("可用的任务: node, link")
    
    while True:
        print("\n选择测试类型:")
        print("1. 单节点解释")
        print("2. 批量解释")
        print("3. 退出")
        
        choice = input("请输入选择 (1-3): ").strip()
        
        if choice == '1':
            dataset = input("数据集: ").strip()
            model = input("模型: ").strip()
            task = input("任务: ").strip()
            node_id = int(input("节点ID: ").strip())
            
            try:
                result = client.explain_single_node(dataset, model, task, node_id)
                print(f"结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
            except Exception as e:
                print(f"失败: {e}")
        
        elif choice == '2':
            dataset = input("数据集: ").strip()
            model = input("模型: ").strip() 
            task = input("任务: ").strip()
            node_ids_str = input("节点ID列表 (用逗号分隔): ").strip()
            node_ids = [int(x.strip()) for x in node_ids_str.split(',')]
            
            try:
                for response_data in client.explain_batch_stream(dataset, model, task, node_ids):
                    print(f"响应: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
            except Exception as e:
                print(f"失败: {e}")
        
        elif choice == '3':
            break
        
        else:
            print("无效选择")

if __name__ == "__main__":
    print("UNR-Explainer API 测试客户端")
    print("=" * 50)
    
    # 基本功能测试
    if not test_basic_functionality():
        print("基本功能测试失败，请检查API服务是否正常运行")
        exit(1)
    
    # 单节点解释测试
    test_single_node_explanation()
    
    # 批量解释测试
    test_batch_explanation()
    
    # 交互式测试
    try:
        interactive_test()
    except KeyboardInterrupt:
        print("\n\n测试结束") 