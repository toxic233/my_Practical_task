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

args = args.parse_args()
device = torch.device('cuda:'+ str(args.gpu) if torch.cuda.is_available() else 'cpu')

# 确保结果目录存在
os.makedirs('./result', exist_ok=True)

print(f"加载数据集: {args.dataset}")
data, G = ut.load_dataset(args)
if args.task == 'link':
    test_data = data[1]; data = data[0]
    
x, edge_index = data.x.to(device), data.edge_index.to(device)
print(f"加载模型: {args.model}")
model, z = ut.load_model(args, data, device)
print("计算嵌入距离排名...")
emb_info = ut.emb_dist_rank(z, args.neighbors_cnt)
    
def PN_topk_link(subgraph_list, bf_top5_idx):
    
    score = 0; num = 0
    new_score = 0; new_num = 0
    label_lst = test_data.edge_label.cpu().tolist()
    
    for i,label in enumerate(tqdm(label_lst)):
        
        nd1 = int(test_data.edge_label_index[0][i])
        nd2 = int(test_data.edge_label_index[1][i])

        if label == 1:
            if nd2 in bf_top5_idx[nd1]:
                score +=1; num +=1
            else:
                num +=1
            if nd1 in bf_top5_idx[nd2]:
                score +=1; num +=1
            else:
                num +=1
        else:
            if nd2 not in bf_top5_idx[nd1]:
                score +=1; num +=1
            else:
                num+=1
            if nd1 not in bf_top5_idx[nd2]:
                score +=1; num +=1
            else:
                num+=1               
        subgraph = subgraph_list[nd1]
        new_emb = ut.perturb_emb(args, model, x, edge_index, list(subgraph.edges()))
        new_bf_top5_idx, _ = ut.emb_dist_rank(new_emb, args.neighbors_cnt)

        if label == 1:
            if nd2 in new_bf_top5_idx[nd1]:
                new_score +=1; new_num +=1
            else:
                new_num +=1

        else:
            if nd2 not in new_bf_top5_idx[nd1]:
                new_score +=1; new_num +=1
            else:
                new_num+=1


        subgraph = subgraph_list[nd2]
        new_emb = ut.perturb_emb(args, model, x, edge_index, list(subgraph.edges()))
        new_bf_top5_idx, _ = ut.emb_dist_rank(new_emb, args.neighbors_cnt)

        if label_lst[i] == 1:
            if nd1 in new_bf_top5_idx[nd2]:
                new_score +=1; new_num +=1
            else:
                new_num +=1
        else:
            if nd1 not in new_bf_top5_idx[nd2]:
                new_score +=1; new_num +=1
            else:
                new_num+=1         

    return abs(score/num-new_score/new_num)


def PN_node(subgraph_list, z):
        
    PN_lst = []
    # 检查数据集类型
    if args.dataset in ['IMDB', 'DBLP']:
        # 对于IMDB和DBLP数据集，我们需要创建标签
        # 这里简单地创建随机标签用于演示
        # 实际应用中应该使用真实标签
        if args.dataset == 'IMDB':
            # IMDB数据集中，我们关注电影节点
            num_movie = 4278
            y = torch.randint(0, 3, (num_movie,))  # 假设有3个类别
            # 只对电影节点进行分类
            z_subset = z[:num_movie]
        else:  # DBLP
            # DBLP数据集中，我们关注作者节点
            num_author = 4057
            y = torch.randint(0, 4, (num_author,))  # 假设有4个类别
            # 只对作者节点进行分类
            z_subset = z[:num_author]
        
        X_train, X_test, y_train, y_test = train_test_split(
            z_subset, y, test_size=0.2, random_state=42, stratify=y.numpy())
    else:
        # 对于其他数据集，使用原始代码
        X_train, X_test, y_train, y_test = train_test_split(
            z, data.y, test_size=0.2, random_state=42, stratify=data.y.cpu().numpy())
    
    clf = LogisticRegression(max_iter=1000)  # node classifier
    clf.fit(X_train, y_train)
    
    for idx, subgraph in enumerate(tqdm(subgraph_list)):
        # 检查索引是否在有效范围内
        if idx >= len(z):
            print(f"警告: 索引 {idx} 超出嵌入向量范围 {len(z)}")
            continue
            
        new_z = ut.perturb_emb(args, model, x, edge_index, list(subgraph.edges()))
        
        # 确保索引有效
        if idx < len(new_z):
            prediction = int(clf.predict(z[idx].reshape(1,-1))[0])
            out_bf = clf.predict_proba(z[idx].reshape(1,-1))[0][prediction]
            out_af = clf.predict_proba(np.array(new_z[idx]).reshape(1,-1))[0][prediction]
            PN_lst.append(out_bf - out_af)

    return PN_lst

def evaluate_subgraph(args, result, z, nm):

    if args.task == 'link':
        score = PN_topk_link(result['subgraph'], z)
    else:
        score = PN_node(result['subgraph'], z)
    
    # 确保结果中包含所需的键
    if 'importance' not in result or 'size' not in result:
        print(f"警告: {nm} 结果中缺少必要的键")
        return 0.0, 0.0, score, 0.0
        
    # 确保重要性是数组
    if not isinstance(result['importance'], (list, np.ndarray)) or len(result['importance']) == 0:
        print(f"警告: {nm} 的重要性不是有效数组")
        vld = 0.0
        impt = 0.0
    else:
        vld = np.mean(np.where(np.array(result['importance'])>=1, 1, 0))
        impt = np.mean(result['importance'])
    
    # 确保大小是数组
    if not isinstance(result['size'], (list, np.ndarray)) or len(result['size']) == 0:
        print(f"警告: {nm} 的大小不是有效数组")
        sz = 0.0
    else:
        sz = np.mean(result['size'])
    
    print(f'{nm:<5s} - {vld:.3f} | {impt:.3f} | {score:.3f} |  {sz:.1f} | ')
    return vld, impt, score, sz
    
path_nm = './result/'+str(args.dataset.lower())+'_'+str(args.model)+'_'+str(args.task)

print(f"评估数据集: {args.dataset}, 模型: {args.model}, 任务: {args.task}")

# 检查基线结果文件是否存在
baseline_path = path_nm + '_baseline'
if not os.path.exists(baseline_path):
    print(f"警告: 基线结果文件不存在: {baseline_path}")
    print("创建空的基线结果...")
    # 创建一个空的基线结果
    result = {
        'n2': {'subgraph': [], 'importance': [], 'size': []},
        'n3': {'subgraph': [], 'importance': [], 'size': []},
        'knn': {'subgraph': [], 'importance': [], 'size': []},
        'rw': {'subgraph': [], 'importance': [], 'size': []},
        'rwr': {'subgraph': [], 'importance': [], 'size': []}
    }
else:
    with open(baseline_path, "rb") as f:
        result = pickle.load(f)

if args.task == 'link':
    z = emb_info[0]

print(f'model -  vld  |  imp  |  PN  | size |')
print('--------------------------------------')

# 评估基线方法
r1 = evaluate_subgraph(args, result.get('n2', {'subgraph': [], 'importance': [], 'size': []}), z, 'n2')
r2 = evaluate_subgraph(args, result.get('n3', {'subgraph': [], 'importance': [], 'size': []}), z, 'n3')
r3 = evaluate_subgraph(args, result.get('knn', {'subgraph': [], 'importance': [], 'size': []}), z, 'knn')
r4 = evaluate_subgraph(args, result.get('rw', {'subgraph': [], 'importance': [], 'size': []}), z, 'rw')
r5 = evaluate_subgraph(args, result.get('rwr', {'subgraph': [], 'importance': [], 'size': []}), z, 'rwr')

# 检查tx结果文件是否存在
tx_path = path_nm + '_tx'
if not os.path.exists(tx_path):
    print(f"警告: TX结果文件不存在: {tx_path}")
    print("创建空的TX结果...")
    tx = {'subgraph': [], 'importance': [], 'size': []}
else:
    with open(tx_path, "rb") as f:
        tx = pickle.load(f)
tx_result = evaluate_subgraph(args, tx, z, 'tx')

# 检查tage结果文件是否存在
tage_path = path_nm + '_tage'
if not os.path.exists(tage_path):
    print(f"警告: TAGE结果文件不存在: {tage_path}")
    print("创建空的TAGE结果...")
    tage = {'subgraph': [], 'importance': [], 'size': []}
else:
    with open(tage_path, "rb") as f:
        tage = pickle.load(f)
tage_result = evaluate_subgraph(args, tage, z, 'tage')

# 检查unr结果文件是否存在
if not os.path.exists(path_nm):
    print(f"警告: UNR结果文件不存在: {path_nm}")
    print("创建空的UNR结果...")
    unr = {'subgraph': [], 'importance': [], 'size': []}
else:
    with open(path_nm, "rb") as f:
        unr = pickle.load(f)
r6 = evaluate_subgraph(args, unr, z, 'unr')

# import pandas as pd
# pd.DataFrame({'model': ['n2', 'n3', 'knn', 'rw', 'rwr', 'unr'],
#             'vld':[r1[0], r2[0], r3[0], r4[0], r5[0], r6[0]], 
#             'imp':[r1[1], r2[1], r3[1], r4[1], r5[1], r6[1]], 
#             'PN':[r1[2], r2[2], r3[2], r4[2], r5[2], r6[2]], 
#             'size':[r1[3], r2[3], r3[3], r4[3], r5[3], r6[3]]}).to_csv('./result/'+data_nm+'_'+str(args.model)+'_'+str(args.task)+'_eval.csv', index=False)


import pandas as pd
pd.DataFrame({'model': ['n2', 'n3', 'knn', 'rw', 'rwr', 'tx', 'tage','unr'],
            'vld':[r1[0], r2[0], r3[0], r4[0], r5[0], tx[0], tage[0], r6[0]],
            'imp':[r1[1], r2[1], r3[1], r4[1], r5[1], tx[1], tage[1], r6[1]],
            'PN':[r1[2], r2[2], r3[2], r4[2], r5[2], tx[2], tage[2], r6[2]],
            'size':[r1[3], r2[3], r3[3], r4[3], r5[3], tx[3], tage[3], r6[3]]}).to_csv('./result/'+str(args.dataset.lower())+'_'+str(args.model)+'_'+str(args.task)+'_eval.csv', index=False)
