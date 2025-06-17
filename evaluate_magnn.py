import time
import argparse
import os

import torch
import torch.nn.functional as F
import numpy as np
import dgl
# 导入MAGNN模块
import sys
sys.path.append('HGNN/MAGNN')
from utils.pytorchtools import EarlyStopping
from utils.data import load_IMDB_data, load_DBLP_data
from utils.tools import evaluate_results_nc, index_generator, parse_minibatch
from model.MAGNN_nc import MAGNN_nc
from model.MAGNN_nc_mb import MAGNN_nc_mb
from HGNN.MAGNN.model import MAGNN_nc, MAGNN_nc_mb

# IMDB和DBLP数据集的参数
IMDB_PARAMS = {
    'out_dim': 3,
    'dropout_rate': 0.5,
    'lr': 0.005,
    'weight_decay': 0.001,
    'etypes_lists': [[[0, 1], [2, 3]],
                    [[1, 0], [1, 2, 3, 0]],
                    [[3, 2], [3, 0, 1, 2]]]
}

DBLP_PARAMS = {
    'out_dim': 4,
    'dropout_rate': 0.5,
    'lr': 0.005,
    'weight_decay': 0.001,
    'etypes_list': [[0, 1], [0, 2, 3, 1], [0, 4, 5, 1]],
    'num_metapaths': 3
}

def evaluate_IMDB(args):
    """在IMDB数据集上评估MAGNN模型性能"""
    print("加载IMDB数据集...")
    nx_G_lists, edge_metapath_indices_lists, features_list, adjM, type_mask, labels, train_val_test_idx = load_IMDB_data()
    
    # 使用GPU如果可用
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"使用设备: {device}")
    
    # 特征类型处理
    features_list = [torch.FloatTensor(features.todense()).to(device) for features in features_list]
    if args.feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif args.feats_type == 1:
        in_dims = [features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(1, len(features_list)):
            features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif args.feats_type == 2:
        in_dims = [features.shape[0] for features in features_list]
        in_dims[0] = features_list[0].shape[1]
        for i in range(1, len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif args.feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    
    # 处理边元路径索引
    edge_metapath_indices_lists = [[torch.LongTensor(indices).to(device) for indices in indices_list] 
                                   for indices_list in edge_metapath_indices_lists]
    
    # 标签和图列表
    labels = torch.LongTensor(labels).to(device)
    g_lists = []
    for nx_G_list in nx_G_lists:
        g_lists.append([])
        for nx_G in nx_G_list:
            g = dgl.DGLGraph(multigraph=True)
            g.add_nodes(nx_G.number_of_nodes())
            g.add_edges(*list(zip(*sorted(map(lambda tup: (int(tup[0]), int(tup[1])), nx_G.edges())))))
            g_lists[-1].append(g)
    
    # 训练、验证和测试索引
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']
    
    # 目标节点索引
    target_node_indices = np.where(type_mask == 0)[0]
    
    # 创建模型
    print("创建模型...")
    net = MAGNN_nc(num_layers=args.layers, 
                 num_metapaths_list=[2, 2, 2], 
                 num_edge_type=4, 
                 etypes_lists=IMDB_PARAMS['etypes_lists'], 
                 feats_dim_list=in_dims, 
                 hidden_dim=args.hidden_dim, 
                 out_dim=IMDB_PARAMS['out_dim'], 
                 num_heads=args.num_heads, 
                 attn_vec_dim=args.attn_vec_dim, 
                 rnn_type=args.rnn_type, 
                 dropout_rate=IMDB_PARAMS['dropout_rate'])
    net.to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=IMDB_PARAMS['lr'], weight_decay=IMDB_PARAMS['weight_decay'])
    
    # 创建checkpoint目录（如果不存在）
    os.makedirs('checkpoint', exist_ok=True)
    
    # 检查是否已有训练好的模型
    save_path = f'checkpoint/checkpoint_IMDB_{args.hidden_dim}_{args.num_heads}.pt'
    if os.path.exists(save_path) and not args.force_train:
        print(f"加载已有模型: {save_path}")
        net.load_state_dict(torch.load(save_path, map_location=device))
    else:
        # 训练模型
        print("训练模型...")
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path=save_path)
        
        t_start = time.time()
        for epoch in range(args.epoch):
            t0 = time.time()
            
            # 训练
            net.train()
            logits, embeddings = net((g_lists, features_list, type_mask, edge_metapath_indices_lists), target_node_indices)
            logp = F.log_softmax(logits, 1)
            train_loss = F.nll_loss(logp[train_idx], labels[train_idx])
            
            # 反向传播
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            # 验证
            net.eval()
            with torch.no_grad():
                logits, embeddings = net((g_lists, features_list, type_mask, edge_metapath_indices_lists), target_node_indices)
                logp = F.log_softmax(logits, 1)
                val_loss = F.nll_loss(logp[val_idx], labels[val_idx])
            
            t1 = time.time()
            
            # 输出训练信息
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Time: {(t1-t0):.4f}s")
            
            # 早停
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break
        
        print(f"训练完成！总耗时: {time.time()-t_start:.2f}s")
    
    # 测试
    print("评估模型...")
    net.eval()
    with torch.no_grad():
        logits, embeddings = net((g_lists, features_list, type_mask, edge_metapath_indices_lists), target_node_indices)
        svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std = evaluate_results_nc(
            embeddings[test_idx].cpu().numpy(), labels[test_idx].cpu().numpy(), num_classes=IMDB_PARAMS['out_dim'])
    
    # 输出结果摘要
    print('----------------------------------------------------------------')
    print('IMDB数据集测试结果：')
    print('SVM测试摘要：')
    print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        macro_f1[0], macro_f1[1], train_size) for macro_f1, train_size in
        zip(svm_macro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
    print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        micro_f1[0], micro_f1[1], train_size) for micro_f1, train_size in
        zip(svm_micro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
    print('K-means测试摘要：')
    print(f'NMI: {nmi_mean:.6f}~{nmi_std:.6f}')
    print(f'ARI: {ari_mean:.6f}~{ari_std:.6f}')


def evaluate_DBLP(args):
    """在DBLP数据集上评估MAGNN模型性能"""
    print("加载DBLP数据集...")
    adjlists, edge_metapath_indices_list, features_list, adjM, type_mask, labels, train_val_test_idx = load_DBLP_data()
    
    # 使用GPU如果可用
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"使用设备: {device}")
    
    # 特征处理
    features_list = [torch.FloatTensor(features).to(device) for features in features_list]
    if args.feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif args.feats_type == 1:
        in_dims = [features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(1, len(features_list)):
            features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif args.feats_type == 2:
        in_dims = [features.shape[0] for features in features_list]
        in_dims[0] = features_list[0].shape[1]
        for i in range(1, len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif args.feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    
    # 标签
    labels = torch.LongTensor(labels).to(device)
    
    # 训练、验证和测试索引
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)
    
    # 创建模型
    print("创建模型...")
    net = MAGNN_nc_mb(
                    DBLP_PARAMS['num_metapaths'],  # 元路径数量
                    6,  # 边类型数量
                    DBLP_PARAMS['etypes_list'],
                    in_dims,
                    args.hidden_dim,
                    DBLP_PARAMS['out_dim'],
                    args.num_heads,
                    args.attn_vec_dim,
                    args.rnn_type,
                    DBLP_PARAMS['dropout_rate'])
    net.to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=DBLP_PARAMS['lr'], weight_decay=DBLP_PARAMS['weight_decay'])
    
    # 创建checkpoint目录（如果不存在）
    os.makedirs('checkpoint', exist_ok=True)
    
    # 检查是否已有训练好的模型
    save_path = f'checkpoint/checkpoint_DBLP_{args.hidden_dim}_{args.num_heads}.pt'
    if os.path.exists(save_path) and not args.force_train:
        print(f"加载已有模型: {save_path}")
        net.load_state_dict(torch.load(save_path, map_location=device))
    else:
        # 训练模型
        print("训练模型...")
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path=save_path)
        
        # 生成batch
        train_idx_generator = index_generator(batch_size=args.batch_size, indices=train_idx)
        val_idx_generator = index_generator(batch_size=args.batch_size, indices=val_idx, shuffle=False)
        
        t_start = time.time()
        for epoch in range(args.epoch):
            t0 = time.time()
            
            # 训练
            net.train()
            train_loss_avg = 0
            for iteration in range(train_idx_generator.num_iterations()):
                # 获取batch数据
                train_idx_batch = train_idx_generator.next()
                train_idx_batch.sort()
                train_g_list, train_indices_list, train_idx_batch_mapped_list = parse_minibatch(
                    adjlists, edge_metapath_indices_list, train_idx_batch, device, args.samples)
                
                # 前向传播
                logits, embeddings = net(
                    (train_g_list, features_list, type_mask, train_indices_list, train_idx_batch_mapped_list))
                logp = F.log_softmax(logits, 1)
                train_loss = F.nll_loss(logp, labels[train_idx_batch])
                train_loss_avg += train_loss.item()
                
                # 反向传播
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                
                # 打印训练信息
                if iteration % 50 == 0:
                    print(f"Epoch {epoch:03d} | Iteration {iteration:03d} | Train Loss: {train_loss.item():.4f}")
            
            train_loss_avg /= train_idx_generator.num_iterations()
            
            # 验证
            net.eval()
            val_loss_avg = 0
            val_logp = []
            with torch.no_grad():
                for iteration in range(val_idx_generator.num_iterations()):
                    val_idx_batch = val_idx_generator.next()
                    val_g_list, val_indices_list, val_idx_batch_mapped_list = parse_minibatch(
                        adjlists, edge_metapath_indices_list, val_idx_batch, device, args.samples)
                    logits, embeddings = net(
                        (val_g_list, features_list, type_mask, val_indices_list, val_idx_batch_mapped_list))
                    logp = F.log_softmax(logits, 1)
                    val_logp.append(logp)
                val_loss = F.nll_loss(torch.cat(val_logp, 0), labels[val_idx])
                val_loss_avg = val_loss.item()
            
            t1 = time.time()
            
            # 输出验证信息
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f} | Time: {(t1-t0):.4f}s")
            
            # 早停
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break
        
        print(f"训练完成！总耗时: {time.time()-t_start:.2f}s")
    
    # 测试
    print("评估模型...")
    test_idx_generator = index_generator(batch_size=args.batch_size, indices=test_idx, shuffle=False)
    net.eval()
    test_embeddings = []
    with torch.no_grad():
        for iteration in range(test_idx_generator.num_iterations()):
            test_idx_batch = test_idx_generator.next()
            test_g_list, test_indices_list, test_idx_batch_mapped_list = parse_minibatch(
                adjlists, edge_metapath_indices_list, test_idx_batch, device, args.samples)
            logits, embeddings = net(
                (test_g_list, features_list, type_mask, test_indices_list, test_idx_batch_mapped_list))
            test_embeddings.append(embeddings)
        test_embeddings = torch.cat(test_embeddings, 0)
        svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std = evaluate_results_nc(
            test_embeddings.cpu().numpy(), labels[test_idx].cpu().numpy(), num_classes=DBLP_PARAMS['out_dim'])
    
    # 输出结果摘要
    print('----------------------------------------------------------------')
    print('DBLP数据集测试结果：')
    print('SVM测试摘要：')
    print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        macro_f1[0], macro_f1[1], train_size) for macro_f1, train_size in
        zip(svm_macro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
    print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        micro_f1[0], micro_f1[1], train_size) for micro_f1, train_size in
        zip(svm_micro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
    print('K-means测试摘要：')
    print(f'NMI: {nmi_mean:.6f}~{nmi_std:.6f}')
    print(f'ARI: {ari_mean:.6f}~{ari_std:.6f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='评估MAGNN模型在IMDB和DBLP数据集上的性能')
    
    # 数据集选择
    ap.add_argument('--dataset', type=str, default='IMDB', choices=['IMDB', 'DBLP', 'both'],
                   help='要评估的数据集（IMDB, DBLP或both）')
    
    # 模型参数
    ap.add_argument('--feats-type', type=int, default=2,
                   help='节点特征类型: ' +
                        '0 - 加载特征; ' +
                        '1 - 仅目标节点特征（其他为零向量）; ' +
                        '2 - 仅目标节点特征（其他为id向量）; ' +
                        '3 - 全部id向量。默认为2。')
    ap.add_argument('--layers', type=int, default=2, help='层数。默认为2。')
    ap.add_argument('--hidden-dim', type=int, default=64, help='节点隐藏状态维度。默认为64。')
    ap.add_argument('--num-heads', type=int, default=8, help='注意力头数量。默认为8。')
    ap.add_argument('--attn-vec-dim', type=int, default=128, help='注意力向量维度。默认为128。')
    ap.add_argument('--rnn-type', default='RotatE0', help='聚合器类型。默认为RotatE0。')
    
    # 训练参数
    ap.add_argument('--epoch', type=int, default=100, help='训练轮数。默认为100。')
    ap.add_argument('--patience', type=int, default=10, help='早停patience。默认为10。')
    ap.add_argument('--batch-size', type=int, default=8, help='批大小，用于DBLP。默认为8。')
    ap.add_argument('--samples', type=int, default=100, help='邻居采样数，用于DBLP。默认为100。')
    ap.add_argument('--force-train', action='store_true', help='强制重新训练，即使有已保存的模型。')
    ap.add_argument('--use-gpu', action='store_true', help='使用GPU（如果可用）。')
    
    args = ap.parse_args()
    
    # 运行评估
    if args.dataset == 'IMDB' or args.dataset == 'both':
        evaluate_IMDB(args)
    
    if args.dataset == 'DBLP' or args.dataset == 'both':
        evaluate_DBLP(args)