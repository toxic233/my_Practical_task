#!/usr/bin/env python3
"""
修复性能瓶颈问题

问题：perturb_emb函数在DBLP数据集处理中每次都重新加载数据，造成严重性能瓶颈
解决方案：使用缓存机制避免重复加载
"""

import sys
import os

def fix_dblp_loading_bottleneck():
    """
    修复DBLP数据重复加载的性能瓶颈
    """
    print("=" * 60)
    print("修复DBLP数据重复加载的性能瓶颈")
    print("=" * 60)
    
    # 读取原文件
    utils_file = 'explainer/utils.py'
    
    with open(utils_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 定位需要替换的部分
    old_section = '''            else:  # DBLP
                # DBLP数据集有4种节点类型
                num_author = 4057
                num_paper = 14328
                num_term = 7723
                num_conf = 20
                
                # 从dataset.magnn_utils.data导入load_DBLP_data函数
                from dataset.magnn_utils.data import load_DBLP_data
                
                # 加载真实的DBLP数据，获取正确维度的特征
                #print("加载DBLP数据以获取正确维度的特征...")
                adjlists, edge_metapath_indices_list, features_list, adjM, type_mask, labels, train_val_test_idx = load_DBLP_data()
                
                # 创建节点特征字典，使用正确维度的特征
                features_0 = torch.FloatTensor(features_list[0])  # author
                features_1 = torch.FloatTensor(features_list[1])  # paper
                features_2 = torch.FloatTensor(features_list[2])  # term
                features_3 = torch.FloatTensor(features_list[3])  # conference
                
                node_features_dict = {
                    'author': features_0,
                    'paper': features_1,
                    'term': features_2,
                    'conference': features_3
                }'''
    
    new_section = '''            else:  # DBLP
                # DBLP数据集有4种节点类型
                num_author = 4057
                num_paper = 14328
                num_term = 7723
                num_conf = 20
                
                # 使用缓存的特征，避免重复加载数据
                if not hasattr(perturb_emb, 'cached_dblp_features'):
                    from dataset.magnn_utils.data import load_DBLP_data
                    print("🔄 首次加载DBLP数据以获取正确维度的特征...")
                    adjlists, edge_metapath_indices_list, features_list, adjM, type_mask, labels, train_val_test_idx = load_DBLP_data()
                    
                    # 缓存特征数据
                    perturb_emb.cached_dblp_features = {
                        'author': torch.FloatTensor(features_list[0]),
                        'paper': torch.FloatTensor(features_list[1]),
                        'term': torch.FloatTensor(features_list[2]),
                        'conference': torch.FloatTensor(features_list[3])
                    }
                    print("✅ DBLP特征数据已缓存，后续调用将直接使用缓存")
                
                # 使用缓存的特征
                node_features_dict = perturb_emb.cached_dblp_features'''
    
    # 检查并替换
    if old_section in content:
        content = content.replace(old_section, new_section)
        print("✅ 找到并替换了DBLP数据加载部分")
    else:
        print("❌ 没有找到完整的DBLP数据加载部分，尝试部分匹配...")
        
        # 尝试更简单的替换
        if "from dataset.magnn_utils.data import load_DBLP_data" in content:
            # 先添加缓存检查
            content = content.replace(
                "# 从dataset.magnn_utils.data导入load_DBLP_data函数\n                from dataset.magnn_utils.data import load_DBLP_data",
                '''# 使用缓存的特征，避免重复加载数据
                if not hasattr(perturb_emb, 'cached_dblp_features'):
                    from dataset.magnn_utils.data import load_DBLP_data'''
            )
            
            # 替换数据加载部分
            content = content.replace(
                "adjlists, edge_metapath_indices_list, features_list, adjM, type_mask, labels, train_val_test_idx = load_DBLP_data()",
                '''print("🔄 首次加载DBLP数据以获取正确维度的特征...")
                    adjlists, edge_metapath_indices_list, features_list, adjM, type_mask, labels, train_val_test_idx = load_DBLP_data()
                    
                    # 缓存特征数据
                    perturb_emb.cached_dblp_features = {
                        'author': torch.FloatTensor(features_list[0]),
                        'paper': torch.FloatTensor(features_list[1]),
                        'term': torch.FloatTensor(features_list[2]),
                        'conference': torch.FloatTensor(features_list[3])
                    }
                    print("✅ DBLP特征数据已缓存，后续调用将直接使用缓存")
                
                # 使用缓存的特征
                node_features_dict = perturb_emb.cached_dblp_features
                
                # 为了兼容性，仍然设置原始变量（但不会重复执行）
                if 'features_list' not in locals():
                    features_list = [
                        node_features_dict['author'].numpy(),
                        node_features_dict['paper'].numpy(),
                        node_features_dict['term'].numpy(),
                        node_features_dict['conference'].numpy()
                    ]'''
            )
            
            # 删除原来的特征创建代码
            old_feature_creation = '''                # 创建节点特征字典，使用正确维度的特征
                features_0 = torch.FloatTensor(features_list[0])  # author
                features_1 = torch.FloatTensor(features_list[1])  # paper
                features_2 = torch.FloatTensor(features_list[2])  # term
                features_3 = torch.FloatTensor(features_list[3])  # conference
                
                node_features_dict = {
                    'author': features_0,
                    'paper': features_1,
                    'term': features_2,
                    'conference': features_3
                }'''
            
            if old_feature_creation in content:
                content = content.replace(old_feature_creation, "")
                print("✅ 删除了原始的特征创建代码")
        
        print("✅ 完成了部分替换")
    
    # 写回文件
    with open(utils_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n" + "=" * 60)
    print("性能瓶颈修复总结")
    print("=" * 60)
    print("🔧 修复的问题:")
    print("   1. 每次MCTS迭代都重新加载DBLP数据集")
    print("   2. 造成大量不必要的I/O和内存操作")
    print("   3. 导致程序长时间卡顿并被killed")
    
    print("\n✅ 解决方案:")
    print("   1. 实现了数据缓存机制")
    print("   2. 首次调用时加载并缓存特征数据")
    print("   3. 后续调用直接使用缓存，避免重复加载")
    print("   4. 显著减少I/O操作和内存分配")
    
    print("\n🚀 预期效果:")
    print("   - 大幅提升MCTS迭代速度")
    print("   - 避免程序被killed的问题")
    print("   - 保持功能的正确性")
    print("   - 减少内存占用")

if __name__ == "__main__":
    fix_dblp_loading_bottleneck() 