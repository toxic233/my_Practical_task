import pickle
import os
import os.path as osp
import datetime
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GraphSAGE, DeepGraphInfomax, SAGEConv
import explainer.args as args
import explainer.utils as ut
import explainer.unrexplainer as unr
import time
from tqdm import tqdm

# 导入MAGNN模型
from fix_magnn_model import MAGNN

def format_time(seconds):
    """将秒数转换为可读的时间格式"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(seconds)}s"

# 创建保存目录
save_dir = "checkpoints"
if not osp.exists(save_dir):
    os.makedirs(save_dir)

args = args.parse_args()
device = torch.device('cuda:'+ str(args.gpu) if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
       
print(f"开始加载数据集: {args.dataset}")
start_time = time.time()
data, G = ut.load_dataset(args)
if args.task == 'link':
    test_data = data[1]; data = data[0]
print(f"数据集加载完成，耗时: {format_time(time.time() - start_time)}")

print(f"开始加载模型: {args.model}")
start_time = time.time()
model, z = ut.load_model(args, data, device)
print(f"模型加载完成，耗时: {format_time(time.time() - start_time)}")

print("计算节点扩展数...")
args.expansion_num = max(int(round(np.mean([G.degree[i] for i in list(G.nodes())]))), 3)
print(f"节点平均扩展数: {args.expansion_num}")

# 根据数据集选择不同的嵌入距离排名函数
print("计算嵌入距离排名...")
start_time = time.time()
if args.dataset == 'DBLP':
    # 对于DBLP数据集，只考虑作者节点作为最近邻
    emb_info = ut.emb_dist_rank_dblp(z, args.neighbors_cnt, author_only=True)
    print("使用DBLP特定的嵌入距离排名函数，只考虑作者节点作为最近邻")
else:
    # 对于其他数据集，使用原始的嵌入距离排名函数
    emb_info = ut.emb_dist_rank(z, args.neighbors_cnt)
print(f"嵌入距离排名计算完成，耗时: {format_time(time.time() - start_time)}")

path_nm = str(args.path)+str(args.dataset.lower())+'_'+str(args.model)+'_'+str(args.task)
expl = dict({'idx':[], 'subgraph':[], 'importance':[], 'size':[], 'time':[]})

# 创建带时间戳的检查点文件名，确保每次运行都是新的
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_file = osp.join(save_dir, f"checkpoint_{args.dataset}_{args.model}_{args.task}_{timestamp}.pkl")

# 创建带时间戳的结果文件名
path_nm_with_timestamp = f"{path_nm}_{timestamp}"

print(f"本次运行将创建新的结果文件: {path_nm_with_timestamp}")
print(f"检查点文件: {checkpoint_file}")

node_lst = list(G.nodes)

# 根据数据集类型选择需要处理的节点
if args.dataset == 'DBLP':
    print("只对作者节点进行元路径扰动")
    # 作者节点的索引范围是0到4056（共4057个节点）
    node_lst = [node for node in node_lst if node < 4057]
    print(f"作者节点数量: {len(node_lst)}")
elif args.dataset == 'IMDB':
    print("只对电影节点进行元路径扰动")
    # 加载IMDB数据集的type_mask来识别电影节点
    from dataset.magnn_utils.data import load_IMDB_data
    try:
        _, _, _, _, type_mask, _, _ = load_IMDB_data()
        # 电影节点的type_mask == 0
        movie_indices = np.where(type_mask == 0)[0]
        node_lst = [node for node in node_lst if node in movie_indices]
        print(f"电影节点数量: {len(node_lst)}")
        print(f"电影节点索引范围: {min(movie_indices)} - {max(movie_indices)}")
    except Exception as e:
        print(f"无法加载IMDB type_mask，处理所有节点: {e}")
        print(f"将处理所有节点数量: {len(node_lst)}")
else:
    print(f"将处理所有节点数量: {len(node_lst)}")

print(f"最终将处理的节点数量: {len(node_lst)}")

# 使用tqdm创建进度条
total_start_time = time.time()
processed_nodes = 0  # 重置为0，因为我们从头开始
save_interval = 10  # 每处理10个节点保存一次

with tqdm(total=len(node_lst), initial=processed_nodes, desc="处理节点") as pbar:
    for i, node in enumerate(node_lst):
        node_start_time = time.time()
        
        # 显示当前处理的节点信息
        pbar.set_description(f"处理节点 {node} ({i+1}/{len(node_lst)})")
        
        subgraph, importance = unr.explainer(args, model, G, data, emb_info, node, device)

        # 计算处理时间
        node_time = time.time() - node_start_time
        total_time = time.time() - total_start_time
        avg_time_per_node = total_time / (i + 1)
        remaining_nodes = len(node_lst) - (i + 1)
        estimated_remaining_time = avg_time_per_node * remaining_nodes

        expl['idx'].append(node)
        expl['time'].append(node_time)
        expl['subgraph'].append(subgraph)
        expl['importance'].append(importance)
        expl['size'].append(subgraph.number_of_edges())
    
        # 更新进度条
        pbar.set_postfix({
            'importance': f"{importance:.4f}",
            'size': subgraph.number_of_edges(),
            'node_time': f"{format_time(node_time)}",
            'est_remaining': f"{format_time(estimated_remaining_time)}"
        })
        pbar.update(1)

        # 定期保存检查点
        if (i + 1) % save_interval == 0:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(expl, f)
            print(f"\n保存检查点到: {checkpoint_file}")
        
        # 保存主结果文件
        with open(path_nm_with_timestamp, "wb") as f:
            pickle.dump(expl, f)

print("\n所有节点处理完成！")
print(f"总处理时间: {format_time(time.time() - total_start_time)}")

# 计算并输出统计信息
if len(expl['importance']) > 0:
    # 计算importance值的统计信息
    importance_values = np.array(expl['importance'])
    importance_mean = np.mean(importance_values)
    importance_max = np.max(importance_values)
    importance_min = np.min(importance_values)
    importance_std = np.std(importance_values)
    
    # 计算size的统计信息
    size_values = np.array(expl['size'])
    size_mean = np.mean(size_values)
    size_max = np.max(size_values)
    size_min = np.min(size_values)
    size_std = np.std(size_values)
    
    print("\n========== 统计信息 ==========")
    print(f"处理的节点数量: {len(expl['idx'])}")
    print(f"Importance值统计:")
    print(f"  - 平均值: {importance_mean:.5f}")
    print(f"  - 最大值: {importance_max:.5f}")
    print(f"  - 最小值: {importance_min:.5f}")
    print(f"  - 标准差: {importance_std:.5f}")
    print(f"Size(子图边数)统计:")
    print(f"  - 平均值: {size_mean:.2f}")
    print(f"  - 最大值: {size_max}")
    print(f"  - 最小值: {size_min}")
    print(f"  - 标准差: {size_std:.2f}")
    print("==============================")

print(f"所有解释已生成并保存到: {path_nm_with_timestamp}")
print(f"检查点文件保存在: {checkpoint_file}")