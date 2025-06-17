import copy
import random
import numpy as np
import math
import networkx as nx
import torch
import explainer.utils as ut

class MCTS:
    def __init__(self, args, G, nd, parent):
        self.args = args
        self.G = G
        self.state = nd
        self.V = 0  
        self.N = 0 
        self.Vi = 0
        self.parent = parent
        self.C = None
        
    def expansion(self, parent):
        n_lst = [n for n in self.G.neighbors(self.state)]
        n_lst_idx = np.random.choice(len(n_lst), min(self.args.expansion_num, len(n_lst)), replace=False)
        n_lst = [n_lst[idx] for idx in n_lst_idx]            
        self.C = {i: MCTS(self.args, self.G, v, parent) for i, v in enumerate(n_lst)}

def UCB1( vi, N, n, c1):
    if n > 0:
        return vi + c1*(math.sqrt(math.log(N)/n))
    else:
        return math.inf
    
def count_subgraph_metapaths(args, model, G, subgraph, initial_nd):
    """
    计算子图中包含初始节点的元路径实例数量
    
    参数:
        args: 参数
        model: 模型
        G: 图
        subgraph: 子图
        initial_nd: 初始节点
        
    返回:
        metapath_count: 子图中的元路径实例数量
    """
    if not hasattr(model, 'metapath_instances'):
        return 0
    
    nd_idx = G.nodes()[initial_nd]['label']
    metapath_count = 0
    subgraph_edges = set(subgraph.edges())
    
    # 对于DBLP数据集，只考虑以作者节点为起点或终点的元路径
    if args.dataset == 'DBLP':
        # 检查初始节点是否为作者节点
        if initial_nd >= 4057:  # 不是作者节点
            return 0
    
    # 遍历所有元路径实例，计算在子图中且包含当前节点的实例数量
    for metapath_type_instances in model.metapath_instances:
        for instance in metapath_type_instances:
            # 检查元路径实例是否包含当前节点
            contains_node = False
            for node_type, node_id in instance:
                # 计算全局节点ID
                if hasattr(model, 'num_author') and hasattr(model, 'num_paper') and hasattr(model, 'num_term'):
                    num_author = model.num_author
                    num_paper = model.num_paper
                    num_term = model.num_term
                    
                    if node_type == 0:  # author
                        global_node_id = node_id
                    elif node_type == 1:  # paper
                        global_node_id = num_author + node_id
                    elif node_type == 2:  # term
                        global_node_id = num_author + num_paper + node_id
                    else:  # conference
                        global_node_id = num_author + num_paper + num_term + node_id
                    
                    if global_node_id == nd_idx:
                        contains_node = True
                        break
            
            # 如果元路径包含当前节点，检查其所有边是否都在子图中
            if contains_node:
                all_edges_in_subgraph = True
                for i in range(len(instance) - 1):
                    node1_type, node1_id = instance[i]
                    node2_type, node2_id = instance[i+1]
                    
                    # 计算全局节点ID
                    if node1_type == 0:  # author
                        global_node1_id = node1_id
                    elif node1_type == 1:  # paper
                        global_node1_id = num_author + node1_id
                    elif node1_type == 2:  # term
                        global_node1_id = num_author + num_paper + node1_id
                    else:  # conference
                        global_node1_id = num_author + num_paper + num_term + node1_id
                        
                    if node2_type == 0:  # author
                        global_node2_id = node2_id
                    elif node2_type == 1:  # paper
                        global_node2_id = num_author + node2_id
                    elif node2_type == 2:  # term
                        global_node2_id = num_author + num_paper + node2_id
                    else:  # conference
                        global_node2_id = num_author + num_paper + num_term + node2_id
                    
                    # 检查边是否在子图中（考虑无向图）
                    edge1 = (global_node1_id, global_node2_id)
                    edge2 = (global_node2_id, global_node1_id)
                    if edge1 not in subgraph_edges and edge2 not in subgraph_edges:
                        all_edges_in_subgraph = False
                        break
                
                if all_edges_in_subgraph:
                    metapath_count += 1
    
    return metapath_count

def select(args, mcts):
    """
    MCT搜索的选择阶段
    
    参数:
        args: 参数
        mcts: MCT搜索树
        
    返回:
        mcts: 更新后的MCT搜索树
        subgraph: 选择的子图
        rw_path: 随机游走路径
    """
    N = mcts.N
    subgraph = reset_subg(mcts.state)
    rw_path = []

    # 添加循环计数器，防止无限循环
    loop_count = 0
    max_loops = 100  # 设置最大循环次数
    
    while mcts.C != None and loop_count < max_loops:
        loop_count += 1
        
        if np.random.rand() < args.restart:
            mcts = reset_agent(mcts)
            rw_path.append(-1)
        else:
            children = mcts.C
            if len(children) == 0:
                mcts = reset_agent(mcts)
                rw_path.append(-1)
                if mcts.parent == None:
                    break
            else:
                try:
                    if (rw_path[-1] == -1) and (len(mcts.C)>=2):
                        s = np.argmax([UCB1(children[i].Vi, N, children[i].N, args.c1) for i in children])
                        nlst = list(range(0, len(mcts.C)))
                        nlst.remove(s)
                        s = np.random.choice(nlst, 1)[0]
                    else:
                        s = np.argmax([UCB1(children[i].Vi, N, children[i].N, args.c1) for i in children])
                        
                except IndexError:
                    s = np.argmax([UCB1(children[i].Vi, N, children[i].N, args.c1) for i in children])
                
                # 添加边到子图，只有当前节点是作者节点时才继续扩展
                if args.dataset == 'DBLP':
                    # 检查当前节点是否为作者节点（索引小于4057）
                    if mcts.state < 4057:
                        subgraph.add_edge(mcts.state, mcts.C[s].state)
                        mcts = mcts.C[s]
                        rw_path.append(s)
                    else:
                        # 如果不是作者节点，重置搜索
                        mcts = reset_agent(mcts)
                        rw_path.append(-1)
                else:
                    # 对于其他数据集，保持原有逻辑
                    subgraph.add_edge(mcts.state, mcts.C[s].state)
                    mcts = mcts.C[s]
                    rw_path.append(s)
    
    # 如果达到最大循环次数，打印警告
    if loop_count >= max_loops:
        print(f"⚠️  select函数达到最大循环次数({max_loops})，强制退出")

    return mcts, subgraph, rw_path

def simulate(args, subgraph, initial_nd, model, G, bf_top5_idx, bf_dist, x, edge_idx):
    """
    使用importance指标进行模拟
    
    参数:
        args: 参数
        subgraph: 子图
        initial_nd: 初始节点
        model: 模型
        G: 图
        bf_top5_idx: 扰动前的最近邻索引
        bf_dist: 扰动前的距离
        x: 节点特征
        edge_idx: 边索引
        
    返回:
        value: importance指标值
    """
    nd_idx = G.nodes()[initial_nd]['label']
    edges_to_perturb = list(subgraph.edges())
    
    # 传递图G参数，用于检查节点的度
    value = ut.importance(args, model, x, edge_idx, bf_top5_idx, bf_dist, edges_to_perturb, nd_idx, G=G)
   
    return value

def simulate_chance(args, subgraph, initial_nd, model, G, x, edge_idx):
    """
    使用chance指标进行模拟
    
    参数:
        args: 参数
        subgraph: 子图
        initial_nd: 初始节点
        model: 模型
        G: 图
        x: 节点特征
        edge_idx: 边索引
        
    返回:
        value: chance指标值
    """
    nd_idx = G.nodes()[initial_nd]['label']
    edges_to_perturb = list(subgraph.edges())
    
    # 获取原始嵌入
    device = torch.device('cuda:'+ str(args.gpu) if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        model.eval()
        if args.model == 'graphsage':
            original_emb = model(x, edge_idx)
        elif args.model == 'dgi':
            original_emb = model.encoder(x, edge_idx)
        elif args.model == 'magnn':
            # 对于MAGNN模型，我们假设已经有了原始嵌入
            # 这里可能需要根据实际情况调整
            if hasattr(model, 'feature_transformers'):
                # 使用特征转换器获取初始嵌入
                original_emb = []
                for node_type, features in model.node_features_dict.items():
                    z_type = model.feature_transformers[node_type](features.to(device))
                    original_emb.append(z_type)
                original_emb = torch.cat(original_emb, dim=0)
            else:
                # 如果没有特征转换器，直接使用前向传播
                original_emb = model(x, edge_idx)
    
    # 确保original_emb在正确的设备上
    if torch.is_tensor(original_emb) and original_emb.device != device:
        original_emb = original_emb.to(device)
    
    # 计算chance指标
    # 传递图G参数，用于检查节点的度
    value = ut.chance(args, model, x, edge_idx, original_emb, edges_to_perturb, nd_idx, G=G)
    
    return value

def backprop(mcts, rw_path, value):
    
    mcts = reset_agent(mcts)
    if value > mcts.V:
        mcts.V = value
        mcts.Vi = mcts.V
    mcts.N += 1

    
    for i in rw_path:
        if i == -1:
            mcts = reset_agent(mcts)
        else:
            mcts = mcts.C[i]
            if value > mcts.V:
                mcts.V = value
                mcts.Vi = mcts.V
            mcts.N += 1

def reset_agent(mcts):
    while mcts.parent != None:
        mcts = mcts.parent
    return mcts

def reset_subg(initial_nd):
    subgraph = nx.Graph()
    subgraph.add_node(initial_nd)
    return subgraph

def khop_sampling(G,initial_nd):
    subgraph = nx.ego_graph(G, initial_nd, radius=2)
    return subgraph




def explainer(args, model, G, data, emb_info, initial_nd, device):
    
    bf_top5_idx, bf_dist = emb_info[0] , emb_info[1]
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    
    # 检查节点的度
    node_degree = G.degree[initial_nd]
    
    # 如果节点是孤立的（度为0），直接返回0
    if node_degree == 0:
        print(f"节点{initial_nd}是孤立节点（度为0），无法解释，返回importance=0")
        # 创建一个只包含初始节点的子图
        subgraph = nx.Graph()
        subgraph.add_node(initial_nd)
        return subgraph, 0.0
    
    mcts = MCTS(args, G, initial_nd, None)
    mcts.expansion(mcts)
    
    impt_vl = []; impt_sbg = []; num_nodes = []
    
    importance = 0; num_iter = 0; patience = 0; argmax_impt = 0
    n_nodes_khop= khop_sampling(G, initial_nd).number_of_nodes()    
    
    while importance < 1.0 and num_iter < args.maxiter: 

        mcts = reset_agent(mcts)
        mcts, subgraph, rw_path = select(args, mcts)

        # expansion condition
        if (mcts.C == None) and (mcts.N > 0):
            mcts.expansion(mcts)
            if len(mcts.C) > 0: 
                subgraph.add_edge(mcts.state, mcts.C[0].state)
                mcts = mcts.C[0]
                rw_path.append(0)
            else:
                if subgraph.number_of_nodes() ==1:
                    break
        else:
            pass

        importance = simulate(args, subgraph, initial_nd, model, G, bf_top5_idx, bf_dist, x, edge_index)

        n_nodes = subgraph.number_of_nodes()
        num_nodes.append(n_nodes)
        backprop(mcts, rw_path, importance)

        impt_vl.append(importance)
        impt_sbg.append(subgraph)
        num_iter += 1

        print('initial node: ', initial_nd, ' | num try: ', num_iter,' | # of nodes: ', n_nodes, ' | importance: ', importance)
        
        # 添加更严格的退出条件
        if n_nodes == 1:
            break
        elif n_nodes_khop == 2:
            break
        elif (n_nodes_khop == 3) and (num_iter > 20):  # 减少3节点情况的迭代限制
            break
        elif num_iter >= args.maxiter:  # 强制退出条件
            print(f"达到最大迭代次数 {args.maxiter}，强制退出")
            break

        if importance > argmax_impt:
            argmax_impt = importance
            patience = 0
        else:
            patience += 1

            # 更早退出，避免过度迭代
            if (patience > 5) and (num_iter > args.maxiter // 4):  # 动态调整patience
                print(f"patience超过5且迭代超过{args.maxiter // 4}次，提前退出")
                break
            elif (patience > 10) and (num_iter > args.maxiter // 2):
                print(f"patience超过10且迭代超过{args.maxiter // 2}次，提前退出")
                break
            
    if n_nodes==1:
        # 如果子图只有一个节点，直接返回0
        print(f"子图只有一个节点（初始节点{initial_nd}），无法解释，返回importance=0")
        return subgraph, 0.0
    else:
        # 原始的选择逻辑
        max_score = max(impt_vl)
        max_lst = np.where(np.array(impt_vl) == max_score)[0]
        min_nodes = min([v for i,v in enumerate(num_nodes) if i in max_lst])
        fn_idx = [i for i,v in enumerate(num_nodes) if v ==min_nodes and i in max_lst][0]
        fn_sbg = impt_sbg[fn_idx]
        fn_score = impt_vl[fn_idx]

        return fn_sbg, fn_score

def explainer_chance(args, model, G, data, initial_nd, device):
    """
    使用chance指标的解释器
    
    参数:
        args: 参数
        model: 模型
        G: 图
        data: 数据
        initial_nd: 初始节点
        device: 设备
        
    返回:
        fn_sbg: 最终子图
        fn_score: 最终分数
    """
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    
    # 检查节点的度
    node_degree = G.degree[initial_nd]
    
    # 如果节点是孤立的（度为0），直接返回0
    if node_degree == 0:
        print(f"节点{initial_nd}是孤立节点（度为0），无法解释，返回chance=0")
        # 创建一个只包含初始节点的子图
        subgraph = nx.Graph()
        subgraph.add_node(initial_nd)
        return subgraph, 0.0
    
    mcts = MCTS(args, G, initial_nd, None)
    mcts.expansion(mcts)
    
    chance_vl = []; chance_sbg = []; num_nodes = []
    
    chance_score = 0; num_iter = 0; patience = 0; argmax_chance = 0
    n_nodes_khop = khop_sampling(G, initial_nd).number_of_nodes()
    
    while chance_score < 1.0 and num_iter < args.maxiter:
        mcts = reset_agent(mcts)
        mcts, subgraph, rw_path = select(args, mcts)

        # expansion condition
        if (mcts.C == None) and (mcts.N > 0):
            mcts.expansion(mcts)
            if len(mcts.C) > 0:
                subgraph.add_edge(mcts.state, mcts.C[0].state)
                mcts = mcts.C[0]
                rw_path.append(0)
            else:
                if subgraph.number_of_nodes() == 1:
                    break
        else:
            pass

        # 使用chance指标进行模拟
        chance_score = simulate_chance(args, subgraph, initial_nd, model, G, x, edge_index)

        n_nodes = subgraph.number_of_nodes()
        num_nodes.append(n_nodes)
        backprop(mcts, rw_path, chance_score)

        chance_vl.append(chance_score)
        chance_sbg.append(subgraph)
        num_iter += 1

        print('initial node: ', initial_nd, ' | num try: ', num_iter,' | # of nodes: ', n_nodes, ' | chance: ', chance_score)
        
        # 添加更严格的退出条件
        if n_nodes == 1:
            break
        elif n_nodes_khop == 2:
            break
        elif (n_nodes_khop == 3) and (num_iter > 20):  # 减少3节点情况的迭代限制
            break
        elif num_iter >= args.maxiter:  # 强制退出条件
            print(f"达到最大迭代次数 {args.maxiter}，强制退出")
            break

        if chance_score > argmax_chance:
            argmax_chance = chance_score
            patience = 0
        else:
            patience += 1

            # 更早退出，避免过度迭代
            if (patience > 5) and (num_iter > args.maxiter // 4):  # 动态调整patience
                print(f"patience超过5且迭代超过{args.maxiter // 4}次，提前退出")
                break
            elif (patience > 10) and (num_iter > args.maxiter // 2):
                print(f"patience超过10且迭代超过{args.maxiter // 2}次，提前退出")
                break
            
    if n_nodes == 1:
        # 如果子图只有一个节点，直接返回0
        print(f"子图只有一个节点（初始节点{initial_nd}），无法解释，返回chance=0")
        return subgraph, 0.0
    else:
        # 原始的选择逻辑
        max_score = max(chance_vl)
        max_lst = np.where(np.array(chance_vl) == max_score)[0]
        min_nodes = min([v for i,v in enumerate(num_nodes) if i in max_lst])
        fn_idx = [i for i,v in enumerate(num_nodes) if v == min_nodes and i in max_lst][0]
        fn_sbg = chance_sbg[fn_idx]
        fn_score = chance_vl[fn_idx]

        return fn_sbg, fn_score