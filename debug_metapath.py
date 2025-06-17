#!/usr/bin/env python3
"""
调试元路径生成和模型加载
"""

import sys
import os
import traceback

# 添加项目路径
sys.path.append('.')
sys.path.append('./explainer')

def test_step_by_step():
    """
    逐步测试每个组件
    """
    print("=" * 50)
    print("逐步调试测试")
    print("=" * 50)
    
    try:
        print("步骤1: 导入模块...")
        import explainer.utils as ut
        print("✅ 模块导入成功")
        
        print("\n步骤2: 创建参数对象...")
        class TestArgs:
            def __init__(self):
                self.dataset = 'DBLP'
                self.model = 'magnn'
                self.task = 'node'
                self.gpu = 0
                self.hidden_dim = 64
                self.neighbors_cnt = 5
                self.expansion_num = 5
                self.restart = 0.1
                self.c1 = 1.0
                self.perturb = 0.0
                self.maxiter = 50
                self.explainer = 'mctsrestart'
                self.path = './result/'
        
        args_obj = TestArgs()
        print("✅ 参数对象创建成功")
        
        print("\n步骤3: 测试数据集加载...")
        data, G = ut.load_dataset(args_obj)
        print(f"✅ 数据集加载成功: 节点={G.number_of_nodes()}, 边={G.number_of_edges()}")
        
        print("\n步骤4: 测试模型加载...")
        import torch
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        model, z = ut.load_model(args_obj, data, device)
        print("✅ 模型加载成功")
        
        # 检查模型属性
        if hasattr(model, 'metapath_instances'):
            print(f"✅ 模型包含元路径实例: {len(model.metapath_instances)} 种类型")
            for i, instances in enumerate(model.metapath_instances):
                print(f"  类型 {i}: {len(instances)} 个实例")
        else:
            print("❌ 模型缺少元路径实例")
        
        print("\n步骤5: 测试嵌入距离计算...")
        emb_info = ut.emb_dist_rank_dblp(z, args_obj.neighbors_cnt, author_only=True)
        print("✅ 嵌入距离计算成功")
        
        print("\n步骤6: 测试元路径扰动...")
        if hasattr(model, 'metapath_instances') and len(model.metapath_instances) > 0:
            # 获取示例元路径
            sample_metapaths = []
            for metapath_type_instances in model.metapath_instances:
                if len(metapath_type_instances) > 0:
                    sample_metapaths.extend(metapath_type_instances[:1])  # 只取1个
            
            if sample_metapaths:
                print(f"测试扰动 {len(sample_metapaths)} 个元路径...")
                perturbed_emb = ut.perturb_metapath_emb(args_obj, model, data.x, data.edge_index, sample_metapaths)
                print("✅ 元路径扰动测试成功")
                
                # 测试重要性计算
                test_node = 0
                print(f"emb_info类型: {type(emb_info)}")
                if isinstance(emb_info, tuple):
                    print(f"emb_info长度: {len(emb_info)}")
                    bf_dist_rank, bf_dist = emb_info
                    bf_top5_idx = bf_dist_rank[test_node]
                    bf_dist_node = bf_dist[test_node]
                else:
                    bf_top5_idx = emb_info[test_node]
                    bf_dist_node = None
                
                importance_score = ut.metapath_importance(
                    args_obj, model, data.x, data.edge_index, 
                    bf_top5_idx, bf_dist_node, sample_metapaths, test_node, G
                )
                print(f"✅ 重要性计算成功: {importance_score:.6f}")
            else:
                print("❌ 没有可用的元路径实例")
        else:
            print("❌ 模型没有元路径实例")
        
        print("\n🎉 所有测试步骤完成!")
        
    except Exception as e:
        print(f"\n❌ 错误发生在: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_step_by_step() 