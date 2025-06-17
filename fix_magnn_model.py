import torch
import torch.nn as nn
import torch.nn.functional as F

class MAGNNLayer(nn.Module):
    """
    MAGNN层实现
    """
    def __init__(self, in_dim, out_dim, num_metapaths, num_edge_types):
        super(MAGNNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_metapaths = num_metapaths
        self.num_edge_types = num_edge_types
        
        # 元路径实例编码器
        self.metapath_encoders = nn.ModuleList()
        for _ in range(num_metapaths):
            self.metapath_encoders.append(nn.Linear(in_dim, out_dim))
        
        # 节点类型特定注意力
        self.attentions = nn.ModuleList()
        for _ in range(num_edge_types):
            self.attentions.append(nn.Linear(2 * out_dim, 1))
        
        # 元路径级别注意力
        self.metapath_attention = nn.Linear(out_dim, 1)
        
        # 其他层
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.activation = nn.ELU()
    
    def forward(self, features_list, metapath_instances_list, edge_types):
        """
        前向传播
        
        参数:
            features_list: 不同类型节点的特征列表
            metapath_instances_list: 元路径实例列表
            edge_types: 每个元路径对应的边类型
            
        返回:
            output: 输出特征
        """
        metapath_outputs = []
        
        # 对每个元路径进行处理
        for i, instances in enumerate(metapath_instances_list):
            # 判断instances是否为一个列表
            if isinstance(instances, list):
                # 如果是列表（多个元路径实例），合并处理
                encoded_instances = []
                for inst in instances:
                    # 提取节点类型特征并编码
                    node_features = [features_list[node_type][node_id] for node_type, node_id in inst]
                    node_features_tensor = torch.stack(node_features)
                    # 编码元路径实例
                    encoded_inst = self.metapath_encoders[i](node_features_tensor)
                    encoded_instances.append(encoded_inst)
                
                # 合并所有编码的实例
                encoded_instances = torch.stack(encoded_instances)
                
                # 计算注意力得分
                edge_type = edge_types[i]
                attention_scores = self.attentions[edge_type](
                    torch.cat([encoded_instances[:, 0], encoded_instances[:, -1]], dim=1)
                )
                attention_weights = F.softmax(self.leaky_relu(attention_scores), dim=0)
                
                # 加权聚合
                metapath_output = torch.sum(attention_weights * encoded_instances[:, -1], dim=0)
                metapath_outputs.append(metapath_output)
            else:
                # 单个元路径实例
                # 编码元路径实例
                encoded_inst = self.metapath_encoders[i](instances)
                metapath_outputs.append(encoded_inst)
        
        # 元路径级别注意力
        if len(metapath_outputs) > 1:
            metapath_outputs_tensor = torch.stack(metapath_outputs)
            metapath_attention_scores = self.metapath_attention(metapath_outputs_tensor).squeeze()
            metapath_attention_weights = F.softmax(self.leaky_relu(metapath_attention_scores), dim=0)
            output = torch.sum(metapath_attention_weights.unsqueeze(1) * metapath_outputs_tensor, dim=0)
        else:
            output = metapath_outputs[0]
        
        return self.activation(output)

class MAGNN(nn.Module):
    """
    MAGNN模型实现
    """
    def __init__(self, node_features_dict, num_nodes_per_type, hidden_dim, num_classes, num_layers=2):
        super(MAGNN, self).__init__()
        
        self.node_features_dict = node_features_dict  # 节点特征字典，key为节点类型
        self.num_nodes_per_type = num_nodes_per_type  # 每种类型的节点数量
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        
        # 节点类型的数量
        self.num_node_types = len(node_features_dict)
        
        # 输入特征维度字典
        self.input_dims = {
            node_type: features.shape[1] for node_type, features in node_features_dict.items()
        }
        
        # 节点类型特定的特征转换
        self.feature_transformers = nn.ModuleDict()
        for node_type, features in node_features_dict.items():
            self.feature_transformers[node_type] = nn.Linear(features.shape[1], hidden_dim)
        
        # MAGNN层
        self.magnn_layers = nn.ModuleList()
        
        # 设置元路径和边类型的数量
        # 元路径: APA, APCPA, APT, PAP, PTP, PCP
        num_metapaths = 6  # 更新为6种元路径
        # 边类型: author-paper, paper-author, paper-term, term-paper, paper-conference, conference-paper
        num_edge_types = 6  # 更新为6种边类型
        
        for _ in range(num_layers):
            self.magnn_layers.append(MAGNNLayer(hidden_dim, hidden_dim, num_metapaths, num_edge_types))
        
        # 分类层
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # 其他层
        self.dropout = nn.Dropout(0.5)
        
        # 动态设置节点数量信息，根据实际的键名
        node_type_keys = list(num_nodes_per_type.keys())
        
        # 为不同数据集设置不同的节点数量属性
        if 'author' in num_nodes_per_type:
            # DBLP数据集
            self.num_author = num_nodes_per_type['author']
            self.num_paper = num_nodes_per_type['paper']
            self.num_term = num_nodes_per_type['term']
            self.num_conf = num_nodes_per_type['conference']
        elif 'movie' in num_nodes_per_type:
            # IMDB数据集
            self.num_movie = num_nodes_per_type['movie']
            self.num_director = num_nodes_per_type['director']
            self.num_actor = num_nodes_per_type['actor']
        else:
            # 其他数据集，动态设置属性
            for node_type, count in num_nodes_per_type.items():
                setattr(self, f'num_{node_type}', count)
        
        # 初始化空的元路径实例列表，将在外部设置
        self.metapath_instances = []
    
    def forward(self, x, edge_index):
        """
        前向传播
        
        参数:
            x: 节点特征
            edge_index: 边索引
            
        返回:
            output: 节点分类的logits
        """
        # 对每种类型的节点进行特征转换
        transformed_features = {}
        for node_type, features in self.node_features_dict.items():
            transformed_features[node_type] = self.feature_transformers[node_type](features)
        
        # 将转换后的特征拼接成一个张量
        h = torch.cat([transformed_features[node_type] for node_type in self.node_features_dict.keys()], dim=0)
        
        # 通过MAGNN层
        for layer in self.magnn_layers:
            h = layer(h, self.metapath_instances, edge_index)
            h = self.dropout(h)
        
        # 分类
        output = self.classifier(h)
        
        return output

def create_magnn_model_for_dblp(node_features_dict, num_nodes_per_type, hidden_dim, num_classes):
    """
    为DBLP数据集创建MAGNN模型
    
    参数:
        node_features_dict: 节点特征字典，key为节点类型
        num_nodes_per_type: 每种类型的节点数量
        hidden_dim: 隐藏层维度
        num_classes: 类别数量
        
    返回:
        model: MAGNN模型
    """
    model = MAGNN(
        node_features_dict=node_features_dict,
        num_nodes_per_type=num_nodes_per_type,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )
    return model

def create_magnn_model_for_imdb(node_features_dict, num_nodes_per_type, hidden_dim, num_classes):
    """
    为IMDB数据集创建MAGNN模型
    
    参数:
        node_features_dict: 节点特征字典，key为节点类型
        num_nodes_per_type: 每种类型的节点数量
        hidden_dim: 隐藏层维度
        num_classes: 类别数量
        
    返回:
        model: MAGNN模型
    """
    model = MAGNN(
        node_features_dict=node_features_dict,
        num_nodes_per_type=num_nodes_per_type,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )
    return model