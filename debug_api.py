#!/usr/bin/env python3
"""
API 详细诊断脚本
用于调试500错误的具体原因
"""

import requests
import json
import traceback

def test_single_request_debug():
    """测试单个请求并获取详细错误信息"""
    
    base_url = "http://localhost:8000"
    
    # 简单的测试用例
    test_params = {
        "dataset": "Cora",
        "model": "graphsage", 
        "task": "node",
        "node_id": 0,
        "timeout": 30
    }
    
    print("🔍 API 详细诊断")
    print("=" * 50)
    
    try:
        print("1. 健康检查...")
        health_response = requests.get(f"{base_url}/api/v1/health", timeout=10)
        print(f"   状态码: {health_response.status_code}")
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"   设备: {health_data.get('device')}")
            print(f"   CUDA可用: {health_data.get('cuda_available')}")
            print(f"   已加载模型: {health_data.get('loaded_models')}")
        else:
            print(f"   健康检查失败: {health_response.text}")
            return
            
        print("\n2. 获取可用模型...")
        models_response = requests.get(f"{base_url}/api/v1/models", timeout=10)
        print(f"   状态码: {models_response.status_code}")
        if models_response.status_code == 200:
            models_data = models_response.json()
            print(f"   支持的数据集: {models_data.get('supported_datasets', [])}")
            print(f"   支持的模型: {models_data.get('supported_model_types', [])}")
            print(f"   已加载模型: {models_data.get('loaded_models', [])}")
        else:
            print(f"   获取模型信息失败: {models_response.text}")
            
        print(f"\n3. 测试单节点解释...")
        print(f"   参数: {test_params}")
        
        # 发送请求并获取详细响应
        response = requests.post(
            f"{base_url}/api/v1/explain/single",
            params=test_params,
            timeout=60
        )
        
        print(f"   状态码: {response.status_code}")
        print(f"   响应头: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ 成功!")
            print(f"   节点ID: {result.get('node_id')}")
            print(f"   重要性: {result.get('importance')}")
            print(f"   处理时间: {result.get('processing_time')}秒")
        else:
            print(f"   ❌ 失败!")
            print(f"   响应内容: {response.text}")
            
            # 尝试解析JSON错误
            try:
                error_data = response.json()
                print(f"   错误详情: {error_data.get('detail', '未知错误')}")
            except:
                print("   无法解析错误JSON")
                
    except requests.exceptions.Timeout:
        print("   ⏰ 请求超时")
    except requests.exceptions.ConnectionError:
        print("   🔌 连接失败，请确认API服务已启动")
    except Exception as e:
        print(f"   ❌ 异常: {e}")
        traceback.print_exc()

def test_import_dependencies():
    """测试项目依赖是否可以正常导入"""
    
    print("\n🔗 测试项目依赖导入")
    print("=" * 50)
    
    dependencies = [
        ("torch", "import torch"),
        ("numpy", "import numpy as np"),
        ("networkx", "import networkx as nx"),
        ("fastapi", "from fastapi import FastAPI"),
        ("pydantic", "from pydantic import BaseModel"),
        ("explainer.args", "import explainer.args as args"),
        ("explainer.utils", "import explainer.utils as ut"),
        ("explainer.unrexplainer", "import explainer.unrexplainer as unr")
    ]
    
    for name, import_cmd in dependencies:
        try:
            exec(import_cmd)
            print(f"   ✅ {name}: 导入成功")
        except Exception as e:
            print(f"   ❌ {name}: 导入失败 - {e}")

def test_direct_model_load():
    """直接测试模型加载功能"""
    
    print("\n🤖 测试模型加载")
    print("=" * 50)
    
    try:
        # 直接导入和测试模型加载相关模块
        import explainer.args as args
        import explainer.utils as ut
        
        print("   ✅ 模块导入成功")
        
        # 创建参数
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', default='Cora')
        parser.add_argument('--model', default='graphsage')
        parser.add_argument('--task', default='node')
        parser.add_argument('--gpu', default='0')
        parser.add_argument('--hidden_dim', default=128, type=int)
        
        test_args = parser.parse_args([])
        print(f"   ✅ 参数创建成功: {test_args.dataset}, {test_args.model}, {test_args.task}")
        
        # 测试数据集加载
        print("   🔄 尝试加载数据集...")
        data, G = ut.load_dataset(test_args)
        print(f"   ✅ 数据集加载成功: {len(G.nodes())} 节点, {len(G.edges())} 边")
        
        # 测试模型加载
        print("   🔄 尝试加载模型...")
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, z = ut.load_model(test_args, data, device)
        print(f"   ✅ 模型加载成功: 嵌入维度 {z.shape}")
        
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        traceback.print_exc()

def check_data_files():
    """检查数据文件是否存在"""
    
    print("\n📁 检查数据文件")
    print("=" * 50)
    
    import os
    
    # 检查常见的数据目录
    data_dirs = [
        'data',
        'dataset', 
        'datasets',
        './data',
        './dataset',
        './datasets'
    ]
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            print(f"   ✅ 找到数据目录: {data_dir}")
            try:
                files = os.listdir(data_dir)
                print(f"      包含文件: {files[:5]}{'...' if len(files) > 5 else ''}")
            except:
                print(f"      无法读取目录内容")
        else:
            print(f"   ❌ 数据目录不存在: {data_dir}")
    
    # 检查模型目录
    model_dirs = [
        'model',
        'models', 
        'checkpoints',
        './model',
        './models',
        './checkpoints'
    ]
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            print(f"   ✅ 找到模型目录: {model_dir}")
            try:
                files = os.listdir(model_dir)
                print(f"      包含文件: {files[:5]}{'...' if len(files) > 5 else ''}")
            except:
                print(f"      无法读取目录内容")
        else:
            print(f"   ❌ 模型目录不存在: {model_dir}")

if __name__ == "__main__":
    print("🚨 UNR-Explainer API 诊断工具")
    print("🔧 用于调试500错误问题")
    print("=" * 60)
    
    try:
        # 1. 测试依赖导入
        test_import_dependencies()
        
        # 2. 检查数据文件
        check_data_files()
        
        # 3. 直接测试模型加载
        test_direct_model_load()
        
        # 4. 测试API请求
        test_single_request_debug()
        
        print("\n" + "=" * 60)
        print("🎯 诊断完成!")
        print("请查看上述输出，找到失败的环节进行修复。")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断诊断")
    except Exception as e:
        print(f"\n❌ 诊断过程中出错: {e}")
        traceback.print_exc() 