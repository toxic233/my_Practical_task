#!/usr/bin/env python3
"""
简化的API功能测试
"""

import sys
sys.path.append('.')

import torch
import argparse
import explainer.args as args
import explainer.utils as ut
import explainer.unrexplainer as unr

def test_direct_explanation():
    """直接测试解释功能，不通过API"""
    
    print("🧪 直接测试解释功能")
    print("=" * 50)
    
    try:
        # 测试参数
        parser = argparse.ArgumentParser()
        
        # 基本参数 - 使用已知存在的模型
        parser.add_argument('--dataset', default='syn1')
        parser.add_argument('--model', default='graphsage')
        parser.add_argument('--task', default='node')
        parser.add_argument('--gpu', default='0')
        parser.add_argument('--hidden_dim', default=128, type=int)
        parser.add_argument('--neighbors_cnt', default=10, type=int)
        parser.add_argument('--num_layers', default=2, type=int)
        
        # 解释器参数
        parser.add_argument('--samples', default=10, type=int)  # 减少样本数以加快测试
        parser.add_argument('--mcts_simulations', default=50, type=int)  # 减少模拟次数
        parser.add_argument('--maxiter', default=20, type=int)  # 减少最大迭代次数
        parser.add_argument('--max_depth', default=2, type=int)  # 减少最大深度
        parser.add_argument('--c_puct', default=5.0, type=float)
        parser.add_argument('--restart', default=0.1, type=float)
        parser.add_argument('--expansion_num', default=3, type=int)
        parser.add_argument('--c1', default=1.0, type=float)
        
        test_args = parser.parse_args([])
        
        print(f"✅ 参数: {test_args.dataset}, {test_args.model}, {test_args.task}")
        
        # 设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ 设备: {device}")
        
        # 加载数据集
        print("🔄 加载数据集...")
        data, G = ut.load_dataset(test_args)
        print(f"✅ 数据集: {len(G.nodes())} 节点, {len(G.edges())} 边")
        
        # 加载模型
        print("🔄 加载模型...")
        model, z = ut.load_model(test_args, data, device)
        print(f"✅ 模型: 嵌入维度 {z.shape}")
        
        # 处理嵌入信息
        print("🔄 处理嵌入信息...")
        emb_info = ut.emb_dist_rank(z, test_args.neighbors_cnt)
        print(f"✅ 嵌入信息处理完成")
        
        # 测试解释
        node_id = 0
        print(f"🔄 解释节点 {node_id}...")
        
        subgraph, importance_score = unr.explainer(
            test_args, model, G, data, emb_info, node_id, device
        )
        
        print(f"✅ 解释完成!")
        print(f"   节点ID: {node_id}")
        print(f"   重要性分数: {importance_score}")
        print(f"   子图节点数: {subgraph.number_of_nodes()}")
        print(f"   子图边数: {subgraph.number_of_edges()}")
        print(f"   子图节点: {list(subgraph.nodes())}")
        
        # 测试另一个节点
        node_id = 1
        print(f"\n🔄 解释节点 {node_id}...")
        
        subgraph2, importance_score2 = unr.explainer(
            test_args, model, G, data, emb_info, node_id, device
        )
        
        print(f"✅ 解释完成!")
        print(f"   节点ID: {node_id}")
        print(f"   重要性分数: {importance_score2}")
        print(f"   子图节点数: {subgraph2.number_of_nodes()}")
        print(f"   子图边数: {subgraph2.number_of_edges()}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_datasets():
    """测试多个数据集"""
    
    print("\n🧪 测试多个数据集")
    print("=" * 50)
    
    # 根据实际可用的模型文件组合
    test_cases = [
        {'dataset': 'syn1', 'model': 'graphsage', 'task': 'node'},
        {'dataset': 'syn3', 'model': 'graphsage', 'task': 'node'},
        {'dataset': 'syn4', 'model': 'graphsage', 'task': 'node'},
        {'dataset': 'PubMed', 'model': 'dgi', 'task': 'node'},
        {'dataset': 'Cora', 'model': 'graphsage', 'task': 'link'},
    ]
    
    success_count = 0
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- 测试用例 {i}: {case['dataset']} + {case['model']} + {case['task']} ---")
        
        try:
            # 创建参数
            parser = argparse.ArgumentParser()
            parser.add_argument('--dataset', default=case['dataset'])
            parser.add_argument('--model', default=case['model'])
            parser.add_argument('--task', default=case['task'])
            parser.add_argument('--gpu', default='0')
            parser.add_argument('--hidden_dim', default=128, type=int)
            parser.add_argument('--neighbors_cnt', default=10, type=int)
            parser.add_argument('--num_layers', default=2, type=int)
            parser.add_argument('--samples', default=5, type=int)
            parser.add_argument('--mcts_simulations', default=30, type=int)
            parser.add_argument('--maxiter', default=10, type=int)
            parser.add_argument('--max_depth', default=2, type=int)
            parser.add_argument('--c_puct', default=5.0, type=float)
            parser.add_argument('--restart', default=0.1, type=float)
            parser.add_argument('--expansion_num', default=3, type=int)
            parser.add_argument('--c1', default=1.0, type=float)
            
            test_args = parser.parse_args([])
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 加载数据和模型
            data, G = ut.load_dataset(test_args)
            model, z = ut.load_model(test_args, data, device)
            
            if case['dataset'].upper() == 'DBLP':
                emb_info = ut.emb_dist_rank_dblp(z, test_args.neighbors_cnt, True)
            else:
                emb_info = ut.emb_dist_rank(z, test_args.neighbors_cnt)
            
            # 测试解释
            node_id = 0
            subgraph, importance_score = unr.explainer(
                test_args, model, G, data, emb_info, node_id, device
            )
            
            print(f"   ✅ 成功! 重要性: {importance_score:.4f}, 子图: {subgraph.number_of_nodes()} 节点")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ 失败: {e}")
    
    print(f"\n🎯 总结: {success_count}/{len(test_cases)} 个测试用例成功")
    return success_count == len(test_cases)

if __name__ == "__main__":
    print("🚀 UNR-Explainer 功能测试")
    print("=" * 60)
    
    # 测试1：基本功能
    success1 = test_direct_explanation()
    
    # 测试2：多数据集
    success2 = test_multiple_datasets()
    
    print(f"\n{'='*60}")
    if success1 and success2:
        print("🎉 所有测试通过！API功能正常")
    else:
        print("❌ 部分测试失败，请检查错误信息")
    print(f"{'='*60}") 