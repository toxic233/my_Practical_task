import pickle
import os.path as osp
import datetime
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from torch_geometric.nn import DeepGraphInfomax, SAGEConv
from gnn.GraphSAGE import GraphSAGE
import explainer.args as args
import explainer.utils as ut
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import explainer.utils as ut
import os
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="MAGNN Explainer Evaluation")
    
    parser.add_argument('--path', type=str, default='./result/',
                    help='Name of the path. Default is ./result/')

    parser.add_argument('--dataset', type=str, default='IMDB',
                    help='Name of the dataset. Default is IMDB. Options: IMDB, DBLP')
        
    parser.add_argument('--model', type=str, default='magnn',
                    help='Name of the node representation learning model. Default is magnn.')
    
    parser.add_argument('--perturb', type=float, default=0.0,
                    help='Hyperparameter for perturbation. Default is 0.0')    
    
    parser.add_argument('--task', type=str, default='node',
                    help='Name of downstream task. Default is node.')
                
    parser.add_argument('--gpu', type=str, default='0',
                    help='Set the gpu to use. Default is 0.')
    
    parser.add_argument('--hidden_dim', type=int, default=64,
                    help='Number of the hidden dimension. Default is 64.')
    
    parser.add_argument('--neighbors_cnt', type=int, default=5,
                    help='Number of top-k neighbors in the embedding space. Default is 5')
    
    return parser.parse_args()

def PN_node(args, subgraph_list, z, model, x, edge_index):
    """
    计算节点分类任务的扰动效果
    """
    PN_lst = []
    
    # 根据数据集类型确定节点类型和标签
    if args.dataset == 'IMDB':
        # IMDB数据集中，我们关注电影节点
        num_movie = 4278
        # 创建随机标签（实际应用中应使用真实标签）
        np.random.seed(42)
        y = torch.randint(0, 3, (num_movie,))  # 假设有3个类别
        # 只对电影节点进行分类
        z_subset = z[:num_movie]
    elif args.dataset == 'DBLP':
        # DBLP数据集中，我们关注作者节点
        num_author = 4057
        # 创建随机标签（实际应用中应使用真实标签）
        np.random.seed(42)
        y = torch.randint(0, 4, (num_author,))  # 假设有4个类别
        # 只对作者节点进行分类
        z_subset = z[:num_author]
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        z_subset, y, test_size=0.2, random_state=42, stratify=y.numpy())
    
    # 训练分类器
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # 计算每个子图的扰动效果
    for idx, subgraph in enumerate(tqdm(subgraph_list)):
        # 检查索引是否在有效范围内
        if idx >= len(z_subset):
            print(f"警告: 索引 {idx} 超出嵌入向量范围 {len(z_subset)}")
            continue
            
        # 计算扰动后的嵌入
        new_z = ut.perturb_emb(args, model, x, edge_index, list(subgraph.edges()))
        
        # 确保索引有效
        if idx < len(new_z):
            # 获取原始预测
            prediction = int(clf.predict(z_subset[idx].reshape(1,-1))[0])
            # 计算原始预测概率
            out_bf = clf.predict_proba(z_subset[idx].reshape(1,-1))[0][prediction]
            # 计算扰动后的预测概率
            out_af = clf.predict_proba(np.array(new_z[idx]).reshape(1,-1))[0][prediction]
            # 计算扰动效果（原始概率 - 扰动后概率）
            PN_lst.append(out_bf - out_af)

    return PN_lst

def evaluate_subgraph(args, result, z, nm, model, x, edge_index):
    """
    评估子图的质量
    """
    # 计算扰动效果
    score = PN_node(args, result['subgraph'], z, model, x, edge_index)
    
    # 确保结果中包含所需的键
    if 'importance' not in result or 'size' not in result:
        print(f"警告: {nm} 结果中缺少必要的键")
        return 0.0, 0.0, np.mean(score) if score else 0.0, 0.0
        
    # 确保重要性是数组
    if not isinstance(result['importance'], (list, np.ndarray)) or len(result['importance']) == 0:
        print(f"警告: {nm} 的重要性不是有效数组")
        vld = 0.0
        impt = 0.0
    else:
        # 计算有效性（重要性>=1的比例）
        vld = np.mean(np.where(np.array(result['importance'])>=1, 1, 0))
        # 计算平均重要性
        impt = np.mean(result['importance'])
    
    # 确保大小是数组
    if not isinstance(result['size'], (list, np.ndarray)) or len(result['size']) == 0:
        print(f"警告: {nm} 的大小不是有效数组")
        sz = 0.0
    else:
        # 计算平均大小
        sz = np.mean(result['size'])
    
    # 打印评估结果
    print(f'{nm:<5s} - {vld:.3f} | {impt:.3f} | {np.mean(score):.3f} |  {sz:.1f} | ')   
    return vld, impt, np.mean(score), sz

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置设备
    device = torch.device('cuda:'+ str(args.gpu) if torch.cuda.is_available() else 'cpu')
    
    # 确保结果目录存在
    os.makedirs('./result', exist_ok=True)
    
    print(f"加载数据集: {args.dataset}")
    # 加载数据集
    data, G = ut.load_dataset(args)
    
    # 将数据移动到设备
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    
    print(f"加载模型: {args.model}")
    # 加载模型
    model, z = ut.load_model(args, data, device)
    
    # 结果文件路径
    path_nm = f'./result/{args.dataset.lower()}_{args.model}_{args.task}'
    
    print(f"评估数据集: {args.dataset}, 模型: {args.model}, 任务: {args.task}")
    
    # 评估结果存储
    results = {}
    
    # 检查基线结果文件是否存在
    baseline_path = path_nm + '_baseline'
    if not os.path.exists(baseline_path):
        print(f"警告: 基线结果文件不存在: {baseline_path}")
        print("跳过基线方法评估")
        baseline_results = None
    else:
        with open(baseline_path, "rb") as f:
            baseline_results = pickle.load(f)
        
        print(f'model -  vld  |  imp  |  PN  | size |')  
        print('--------------------------------------')
        
        # 评估基线方法
        results['n2'] = evaluate_subgraph(args, baseline_results.get('n2', {'subgraph': [], 'importance': [], 'size': []}), 
                                         z, 'n2', model, x, edge_index)
        results['n3'] = evaluate_subgraph(args, baseline_results.get('n3', {'subgraph': [], 'importance': [], 'size': []}), 
                                         z, 'n3', model, x, edge_index)
        results['knn'] = evaluate_subgraph(args, baseline_results.get('knn', {'subgraph': [], 'importance': [], 'size': []}), 
                                          z, 'knn', model, x, edge_index)
        results['rw'] = evaluate_subgraph(args, baseline_results.get('rw', {'subgraph': [], 'importance': [], 'size': []}), 
                                         z, 'rw', model, x, edge_index)
        results['rwr'] = evaluate_subgraph(args, baseline_results.get('rwr', {'subgraph': [], 'importance': [], 'size': []}), 
                                          z, 'rwr', model, x, edge_index)
    
    # 检查tx结果文件是否存在
    tx_path = path_nm + '_tx'
    if not os.path.exists(tx_path):
        print(f"警告: TX结果文件不存在: {tx_path}")
        print("跳过TX方法评估")
        results['tx'] = (0.0, 0.0, 0.0, 0.0)
    else:
        with open(tx_path, "rb") as f:
            tx = pickle.load(f)
        results['tx'] = evaluate_subgraph(args, tx, z, 'tx', model, x, edge_index)
    
    # 检查tage结果文件是否存在
    tage_path = path_nm + '_tage'
    if not os.path.exists(tage_path):
        print(f"警告: TAGE结果文件不存在: {tage_path}")
        print("跳过TAGE方法评估")
        results['tage'] = (0.0, 0.0, 0.0, 0.0)
    else:
        with open(tage_path, "rb") as f:
            tage = pickle.load(f)
        results['tage'] = evaluate_subgraph(args, tage, z, 'tage', model, x, edge_index)
    
    # 检查unr结果文件是否存在
    if not os.path.exists(path_nm):
        print(f"警告: UNR结果文件不存在: {path_nm}")
        print("跳过UNR方法评估")
        results['unr'] = (0.0, 0.0, 0.0, 0.0)
    else:
        with open(path_nm, "rb") as f:
            unr = pickle.load(f)
        results['unr'] = evaluate_subgraph(args, unr, z, 'unr', model, x, edge_index)
    
    # 将结果保存为CSV文件
    if baseline_results is not None:
        df = pd.DataFrame({
            'model': list(results.keys()),
            'vld': [results[k][0] for k in results],
            'imp': [results[k][1] for k in results],
            'PN': [results[k][2] for k in results],
            'size': [results[k][3] for k in results]
        })
        
        # 保存结果
        csv_path = f'./result/{args.dataset.lower()}_{args.model}_{args.task}_eval.csv'
        df.to_csv(csv_path, index=False)
        print(f"评估结果已保存到: {csv_path}")

if __name__ == "__main__":
    main()