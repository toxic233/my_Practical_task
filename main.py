import asyncio
import time
import traceback
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import json
import logging
from datetime import datetime

# 导入项目模块
import explainer.args as args
import explainer.utils as ut
import explainer.unrexplainer as unr

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局模型管理器
class ModelManager:
    def __init__(self):
        self.models = {}
        self.data_cache = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"初始化ModelManager，使用设备: {self.device}")
    
    async def load_model(self, dataset: str, model_name: str, task: str):
        """异步加载模型和数据"""
        cache_key = f"{dataset}_{model_name}_{task}"
        
        if cache_key in self.models:
            logger.info(f"从缓存中获取模型: {cache_key}")
            return self.models[cache_key], self.data_cache[cache_key]
        
        try:
            logger.info(f"开始加载模型: {cache_key}")
            start_time = time.time()
            
            # 创建args对象
            import argparse
            parser = argparse.ArgumentParser()
            # 添加所有必要的参数
            parser.add_argument('--dataset', default=dataset)
            parser.add_argument('--model', default=model_name)
            parser.add_argument('--task', default=task)
            parser.add_argument('--gpu', default='0')
            parser.add_argument('--hidden_dim', default=128, type=int)
            parser.add_argument('--neighbors_cnt', default=5, type=int)
            parser.add_argument('--maxiter', default=1000, type=int)
            parser.add_argument('--c1', default=1.0, type=float)
            parser.add_argument('--restart', default=0.2, type=float)
            parser.add_argument('--perturb', default=0.0, type=float)
            parser.add_argument('--path', default='./result/')
            parser.add_argument('--explainer', default='mctsrestart')
            parser.add_argument('--iter', default=300, type=int)
            parser.add_argument('--lr', default=0.001, type=float)
            
            model_args = parser.parse_args([])
            
            # 加载数据集
            data, G = await asyncio.to_thread(ut.load_dataset, model_args)
            if task == 'link':
                test_data = data[1]
                data = data[0]
            else:
                test_data = None
            
            # 加载模型
            model, z = await asyncio.to_thread(ut.load_model, model_args, data, self.device)
            
            # 计算嵌入距离排名
            if dataset == 'DBLP':
                emb_info = await asyncio.to_thread(ut.emb_dist_rank_dblp, z, model_args.neighbors_cnt, True)
            else:
                emb_info = await asyncio.to_thread(ut.emb_dist_rank, z, model_args.neighbors_cnt)
            
            # 计算节点扩展数
            expansion_num = max(int(round(np.mean([G.degree[i] for i in list(G.nodes())]))), 3)
            model_args.expansion_num = expansion_num
            
            model_data = {
                'model': model,
                'z': z,
                'G': G,
                'data': data,
                'test_data': test_data,
                'emb_info': emb_info,
                'args': model_args
            }
            
            self.models[cache_key] = model_data
            self.data_cache[cache_key] = {
                'dataset': dataset,
                'model_name': model_name,
                'task': task,
                'num_nodes': len(G.nodes()),
                'num_edges': len(G.edges()),
                'device': str(self.device),
                'load_time': time.time() - start_time
            }
            
            logger.info(f"模型加载完成: {cache_key}, 耗时: {time.time() - start_time:.2f}秒")
            return model_data, self.data_cache[cache_key]
            
        except Exception as e:
            logger.error(f"模型加载失败: {cache_key}, 错误: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")
    
    def get_available_models(self):
        """获取已加载的模型列表"""
        return list(self.models.keys())

# 创建全局模型管理器实例
model_manager = ModelManager()

# 参数标准化函数
def normalize_dataset_name(dataset: str) -> str:
    """标准化数据集名称（大小写不敏感）"""
    dataset_mapping = {
        'cora': 'Cora',
        'citeseer': 'CiteSeer', 
        'pubmed': 'PubMed',
        'acm': 'ACM',
        'imdb': 'IMDB',
        'dblp': 'DBLP',
        'syn1': 'syn1',
        'syn2': 'syn2', 
        'syn3': 'syn3',
        'syn4': 'syn4'
    }
    
    normalized = dataset_mapping.get(dataset.lower())
    if normalized is None:
        raise HTTPException(
            status_code=400, 
            detail=f"不支持的数据集 '{dataset}'。支持的数据集: {list(dataset_mapping.values())}"
        )
    return normalized

def normalize_model_name(model: str) -> str:
    """标准化模型名称（大小写不敏感）"""
    model_mapping = {
        'graphsage': 'graphsage',
        'dgi': 'dgi',
        'magnn': 'magnn'
    }
    
    normalized = model_mapping.get(model.lower())
    if normalized is None:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的模型 '{model}'。支持的模型: {list(model_mapping.values())}"
        )
    return normalized

def normalize_task_name(task: str) -> str:
    """标准化任务名称（大小写不敏感）"""
    task_mapping = {
        'node': 'node',
        'link': 'link'
    }
    
    normalized = task_mapping.get(task.lower())
    if normalized is None:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的任务 '{task}'。支持的任务: {list(task_mapping.values())}"
        )
    return normalized

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("FastAPI应用启动")
    yield
    logger.info("FastAPI应用关闭")

# 创建FastAPI应用
app = FastAPI(
    title="UNR-Explainer API",
    description="图神经网络可解释性API服务",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic模型定义
class ExplainRequest(BaseModel):
    dataset: str = Field(..., description="数据集名称")
    model: str = Field(..., description="模型名称")
    task: str = Field(..., description="任务类型")
    node_ids: List[int] = Field(..., description="要解释的节点ID列表")
    neighbors_cnt: Optional[int] = Field(5, description="最近邻节点数量")
    maxiter: Optional[int] = Field(1000, description="MCTS最大迭代次数")
    c1: Optional[float] = Field(1.0, description="探索参数")
    restart: Optional[float] = Field(0.2, description="重启概率")
    perturb: Optional[float] = Field(0.0, description="扰动参数")
    timeout: Optional[int] = Field(300, description="超时时间(秒)")
    
    @validator('dataset')
    def validate_dataset(cls, v):
        # 数据集名称映射表（支持大小写不敏感）
        dataset_mapping = {
            'cora': 'Cora',
            'citeseer': 'CiteSeer', 
            'pubmed': 'PubMed',
            'acm': 'ACM',
            'imdb': 'IMDB',
            'dblp': 'DBLP',
            'syn1': 'syn1',
            'syn2': 'syn2', 
            'syn3': 'syn3',
            'syn4': 'syn4'
        }
        
        # 转换为小写进行匹配
        v_lower = v.lower()
        if v_lower in dataset_mapping:
            return dataset_mapping[v_lower]
        
        valid_datasets = list(dataset_mapping.values())
        raise ValueError(f'数据集必须是以下之一: {valid_datasets} (大小写不敏感)')
        return v
    
    @validator('model')
    def validate_model(cls, v):
        model_mapping = {
            'graphsage': 'graphsage',
            'dgi': 'dgi', 
            'magnn': 'magnn'
        }
        
        v_lower = v.lower()
        if v_lower in model_mapping:
            return model_mapping[v_lower]
            
        valid_models = list(model_mapping.values())
        raise ValueError(f'模型必须是以下之一: {valid_models} (大小写不敏感)')
        return v
    
    @validator('task')
    def validate_task(cls, v):
        task_mapping = {
            'node': 'node',
            'link': 'link'
        }
        
        v_lower = v.lower()
        if v_lower in task_mapping:
            return task_mapping[v_lower]
            
        valid_tasks = list(task_mapping.values())
        raise ValueError(f'任务必须是以下之一: {valid_tasks} (大小写不敏感)')
        return v

class ExplainResult(BaseModel):
    node_id: int
    importance: float
    subgraph_size: int
    processing_time: float
    status: str = "success"
    error: Optional[str] = None

class BatchExplainResponse(BaseModel):
    request_id: str
    total_nodes: int
    processed_nodes: int
    results: List[ExplainResult]
    overall_stats: Dict[str, Any]
    processing_time: float
    status: str

# API端点
@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "UNR-Explainer API服务",
        "version": "1.0.0",
        "device": str(model_manager.device),
        "available_models": model_manager.get_available_models()
    }

@app.get("/api/v1/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "device": str(model_manager.device),
        "cuda_available": torch.cuda.is_available(),
        "loaded_models": len(model_manager.models)
    }

@app.get("/api/v1/models")
async def get_available_models():
    """获取可用模型列表"""
    return {
        "loaded_models": model_manager.get_available_models(),
        "supported_datasets": ['Cora', 'CiteSeer', 'PubMed', 'ACM', 'IMDB', 'DBLP', 'syn1', 'syn2', 'syn3', 'syn4'],
        "supported_model_types": ['graphsage', 'dgi', 'magnn'],
        "supported_tasks": ['node', 'link']
    }

@app.post("/api/v1/explain/batch")
async def explain_batch(request: ExplainRequest, background_tasks: BackgroundTasks):
    """批量解释节点（实时流式响应）"""
    
    async def generate_explanations():
        request_id = f"batch_{int(time.time() * 1000)}"
        start_time = time.time()
        
        try:
            # 加载模型和数据
            yield f"data: {json.dumps({'type': 'status', 'message': '正在加载模型...', 'progress': 0})}\n\n"
            
            model_data, cache_info = await model_manager.load_model(
                request.dataset, request.model, request.task
            )
            
            model = model_data['model']
            G = model_data['G']
            data = model_data['data']
            emb_info = model_data['emb_info']
            model_args = model_data['args']
            
            # 更新参数
            model_args.neighbors_cnt = request.neighbors_cnt
            model_args.maxiter = request.maxiter
            model_args.c1 = request.c1
            model_args.restart = request.restart
            model_args.perturb = request.perturb
            
            yield f"data: {json.dumps({'type': 'status', 'message': '模型加载完成，开始处理节点...', 'progress': 10})}\n\n"
            
            # 过滤有效节点
            valid_node_ids = []
            all_nodes = set(G.nodes())
            
            for node_id in request.node_ids:
                if node_id in all_nodes:
                    # 对于DBLP数据集，只处理作者节点
                    if request.dataset == 'DBLP' and node_id >= 4057:
                        continue
                    # 对于IMDB数据集，只处理电影节点
                    elif request.dataset == 'IMDB':
                        try:
                            from dataset.magnn_utils.data import load_IMDB_data
                            _, _, _, _, type_mask, _, _ = load_IMDB_data()
                            movie_indices = np.where(type_mask == 0)[0]
                            if node_id not in movie_indices:
                                continue
                        except:
                            pass
                    valid_node_ids.append(node_id)
            
            if not valid_node_ids:
                yield f"data: {json.dumps({'type': 'error', 'message': '没有有效的节点ID'})}\n\n"
                return
            
            total_nodes = len(valid_node_ids)
            processed_nodes = 0
            results = []
            
            # 处理每个节点
            for i, node_id in enumerate(valid_node_ids):
                try:
                    node_start_time = time.time()
                    
                    # 设置超时
                    explanation_task = asyncio.create_task(
                        asyncio.to_thread(
                            unr.explainer,
                            model_args, model, G, data, emb_info, node_id, model_manager.device
                        )
                    )
                    
                    try:
                        subgraph, importance = await asyncio.wait_for(
                            explanation_task, timeout=request.timeout
                        )
                        
                        processing_time = time.time() - node_start_time
                        
                        result = ExplainResult(
                            node_id=node_id,
                            importance=float(importance),
                            subgraph_size=subgraph.number_of_edges(),
                            processing_time=processing_time,
                            status="success"
                        )
                        
                        results.append(result)
                        processed_nodes += 1
                        
                        # 实时返回进度
                        progress = int(10 + (i + 1) / total_nodes * 85)
                        yield f"data: {json.dumps({'type': 'progress', 'node_id': node_id, 'result': result.dict(), 'progress': progress, 'processed': processed_nodes, 'total': total_nodes})}\n\n"
                        
                    except asyncio.TimeoutError:
                        result = ExplainResult(
                            node_id=node_id,
                            importance=0.0,
                            subgraph_size=0,
                            processing_time=request.timeout,
                            status="timeout",
                            error=f"处理超时({request.timeout}秒)"
                        )
                        results.append(result)
                        yield f"data: {json.dumps({'type': 'timeout', 'node_id': node_id, 'error': result.error})}\n\n"
                        
                except Exception as e:
                    error_msg = f"处理节点{node_id}时出错: {str(e)}"
                    logger.error(error_msg)
                    result = ExplainResult(
                        node_id=node_id,
                        importance=0.0,
                        subgraph_size=0,
                        processing_time=0.0,
                        status="error",
                        error=str(e)
                    )
                    results.append(result)
                    yield f"data: {json.dumps({'type': 'node_error', 'node_id': node_id, 'error': str(e)})}\n\n"
            
            # 计算总体统计
            successful_results = [r for r in results if r.status == "success"]
            if successful_results:
                importances = [r.importance for r in successful_results]
                sizes = [r.subgraph_size for r in successful_results]
                times = [r.processing_time for r in successful_results]
                
                overall_stats = {
                    "importance_mean": float(np.mean(importances)),
                    "importance_std": float(np.std(importances)),
                    "importance_max": float(np.max(importances)),
                    "importance_min": float(np.min(importances)),
                    "size_mean": float(np.mean(sizes)),
                    "size_std": float(np.std(sizes)),
                    "size_max": int(np.max(sizes)),
                    "size_min": int(np.min(sizes)),
                    "avg_processing_time": float(np.mean(times)),
                    "success_rate": len(successful_results) / len(results)
                }
            else:
                overall_stats = {}
            
            # 返回最终结果
            total_time = time.time() - start_time
            final_response = BatchExplainResponse(
                request_id=request_id,
                total_nodes=total_nodes,
                processed_nodes=processed_nodes,
                results=results,
                overall_stats=overall_stats,
                processing_time=total_time,
                status="completed"
            )
            
            yield f"data: {json.dumps({'type': 'completed', 'final_result': final_response.dict(), 'progress': 100})}\n\n"
            
        except Exception as e:
            error_msg = f"批量处理失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
    
    return StreamingResponse(
        generate_explanations(),
        media_type="text/plain",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

@app.post("/api/v1/explain/single")
async def explain_single(
    dataset: str,
    model: str,
    task: str,
    node_id: int,
    neighbors_cnt: int = 5,
    maxiter: int = 1000,
    c1: float = 1.0,
    restart: float = 0.2,
    perturb: float = 0.0,
    timeout: int = 60
):
    """单个节点解释"""
    
    try:
        start_time = time.time()
        
        # 标准化参数（大小写不敏感）
        dataset = normalize_dataset_name(dataset)
        model = normalize_model_name(model)
        task = normalize_task_name(task)
        
        # 加载模型
        model_data, cache_info = await model_manager.load_model(dataset, model, task)
        
        model_obj = model_data['model']
        G = model_data['G']
        data = model_data['data']
        emb_info = model_data['emb_info']
        model_args = model_data['args']
        
        # 检查节点是否有效
        if node_id not in G.nodes():
            raise HTTPException(status_code=400, detail=f"节点ID {node_id} 不存在")
        
        # 对于特定数据集的节点类型检查
        if dataset == 'DBLP' and node_id >= 4057:
            raise HTTPException(status_code=400, detail=f"DBLP数据集只支持作者节点(ID < 4057)")
        
        # 更新参数
        model_args.neighbors_cnt = neighbors_cnt
        model_args.maxiter = maxiter
        model_args.c1 = c1
        model_args.restart = restart
        model_args.perturb = perturb
        
        # 执行解释
        explanation_task = asyncio.create_task(
            asyncio.to_thread(
                unr.explainer,
                model_args, model_obj, G, data, emb_info, node_id, model_manager.device
            )
        )
        
        try:
            subgraph, importance = await asyncio.wait_for(explanation_task, timeout=timeout)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail=f"处理超时({timeout}秒)")
        
        processing_time = time.time() - start_time
        
        return {
            "node_id": node_id,
            "importance": float(importance),
            "subgraph_size": subgraph.number_of_edges(),
            "subgraph_nodes": list(subgraph.nodes()),
            "subgraph_edges": list(subgraph.edges()),
            "processing_time": processing_time,
            "model_info": cache_info,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"单节点解释失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"解释失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 