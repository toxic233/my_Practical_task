#!/usr/bin/env python3
"""
UNR-Explainer FastAPI 服务器
Python 3.8 兼容版本
"""

import asyncio
import concurrent.futures
import json
import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Any
import argparse

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入项目模块
try:
    import explainer.args as args
    import explainer.utils as ut
    import explainer.unrexplainer as unr
except ImportError as e:
    logger.error(f"导入项目模块失败: {e}")
    raise


class ModelManager:
    """模型管理器，负责模型的加载、缓存和管理"""
    
    def __init__(self, max_cache_size: int = 5):
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"ModelManager 初始化，设备: {self.device}")
        
        # 创建线程池用于异步处理
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
    async def _run_in_thread(self, func, *args, **kwargs):
        """在线程池中运行函数，Python 3.8兼容版本"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args, **kwargs)
    
    def _load_model_sync(self, dataset: str, model: str, task: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """同步加载模型的内部方法"""
        # 创建参数对象
        model_args = self._create_args(dataset, model, task)
        
        try:
            # 加载数据集
            logger.info(f"加载数据集: {dataset}")
            data, G = ut.load_dataset(model_args)
            logger.info(f"数据集加载成功: {len(G.nodes())} 节点, {len(G.edges())} 边")
            
            # 加载模型
            logger.info(f"加载模型: {model}")
            model_obj, z = ut.load_model(model_args, data, self.device)
            logger.info(f"模型加载成功: 嵌入维度 {z.shape}")
            
            # 处理嵌入信息
            if dataset.upper() == 'DBLP':
                emb_info = ut.emb_dist_rank_dblp(z, model_args.neighbors_cnt, True)
            else:
                emb_info = ut.emb_dist_rank(z, model_args.neighbors_cnt)
            
            # 计算节点扩展数
            expansion_num = max(int(round(np.mean([G.degree[i] for i in list(G.nodes())]))), 3)
            model_args.expansion_num = expansion_num
            
            model_data = {
                'args': model_args,
                'data': data,
                'graph': G,
                'model': model_obj,
                'embeddings': z,
                'emb_info': emb_info,
                'device': self.device,
                'load_time': time.time()
            }
            
            cache_info = {
                'cached': False,
                'cache_size': len(self.cache)
            }
            
            return model_data, cache_info
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")
    
    async def load_model(self, dataset: str, model: str, task: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """加载模型（异步）"""
        cache_key = f"{dataset.upper()}_{model}_{task}"
        
        # 检查缓存
        if cache_key in self.cache:
            logger.info(f"从缓存加载模型: {cache_key}")
            cache_info = {
                'cached': True,
                'cache_size': len(self.cache)
            }
            return self.cache[cache_key], cache_info
        
        # 异步加载模型
        model_data, cache_info = await self._run_in_thread(
            self._load_model_sync, dataset, model, task
        )
        
        # 更新缓存
        self._update_cache(cache_key, model_data)
        
        return model_data, cache_info
    
    def _create_args(self, dataset: str, model: str, task: str) -> argparse.Namespace:
        """创建参数对象"""
        # 参数规范化
        dataset = self.normalize_dataset(dataset)
        model = self.normalize_model(model)
        task = self.normalize_task(task)
        
        # 创建参数解析器
        parser = argparse.ArgumentParser()
        
        # 基本参数
        parser.add_argument('--dataset', default=dataset)
        parser.add_argument('--model', default=model)
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
        
        # 数据集特定参数
        if dataset == 'syn1':
            parser.add_argument('--num_layers', default=2, type=int)
        elif dataset == 'syn2':
            parser.add_argument('--num_layers', default=3, type=int)
        else:
            parser.add_argument('--num_layers', default=2, type=int)
        
        # 解释器参数
        parser.add_argument('--samples', default=50, type=int)
        parser.add_argument('--mcts_simulations', default=200, type=int)
        parser.add_argument('--max_depth', default=3, type=int)
        parser.add_argument('--c_puct', default=5.0, type=float)
        
        return parser.parse_args([])
    
    def _update_cache(self, cache_key: str, model_data: Dict[str, Any]):
        """更新缓存"""
        if len(self.cache) >= self.max_cache_size:
            # 移除最旧的缓存项
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k]['load_time'])
            del self.cache[oldest_key]
            logger.info(f"缓存已满，移除: {oldest_key}")
        
        self.cache[cache_key] = model_data
        logger.info(f"缓存更新: {cache_key}")
    
    @staticmethod
    def normalize_dataset(dataset: str) -> str:
        """规范化数据集名称（大小写不敏感）"""
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
        return dataset_mapping.get(dataset.lower(), dataset)
    
    @staticmethod
    def normalize_model(model: str) -> str:
        """规范化模型名称（大小写不敏感）"""
        model_mapping = {
            'graphsage': 'graphsage',
            'dgi': 'dgi',
            'magnn': 'magnn'
        }
        return model_mapping.get(model.lower(), model)
    
    @staticmethod
    def normalize_task(task: str) -> str:
        """规范化任务名称（大小写不敏感）"""
        task_mapping = {
            'node': 'node',
            'link': 'link'
        }
        return task_mapping.get(task.lower(), task)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        return {
            'loaded_models': list(self.cache.keys()),
            'cache_size': len(self.cache),
            'max_cache_size': self.max_cache_size,
            'device': str(self.device)
        }


# 全局模型管理器
model_manager = ModelManager()

# FastAPI 应用
app = FastAPI(
    title="UNR-Explainer API",
    description="图神经网络解释性分析API",
    version="1.0.0"
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 响应模型
class ExplanationResponse(BaseModel):
    node_id: int
    importance: List[float]
    processing_time: float
    model_info: Dict[str, Any]
    cache_info: Dict[str, Any]


class BatchExplanationRequest(BaseModel):
    dataset: str
    model: str
    task: str
    node_ids: List[int]
    timeout: Optional[int] = 60
    
    @validator('dataset')
    def validate_dataset(cls, v):
        valid_datasets = ['Cora', 'CiteSeer', 'PubMed', 'ACM', 'IMDB', 'DBLP', 'syn1', 'syn2', 'syn3', 'syn4']
        normalized = ModelManager.normalize_dataset(v)
        if normalized not in valid_datasets:
            raise ValueError(f'数据集必须是: {valid_datasets}')
        return normalized
    
    @validator('model')
    def validate_model(cls, v):
        valid_models = ['graphsage', 'dgi', 'magnn']
        normalized = ModelManager.normalize_model(v)
        if normalized not in valid_models:
            raise ValueError(f'模型必须是: {valid_models}')
        return normalized
    
    @validator('task')
    def validate_task(cls, v):
        valid_tasks = ['node', 'link']
        normalized = ModelManager.normalize_task(v)
        if normalized not in valid_tasks:
            raise ValueError(f'任务必须是: {valid_tasks}')
        return normalized


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    device: str
    cuda_available: bool
    loaded_models: int


class ModelsResponse(BaseModel):
    supported_datasets: List[str]
    supported_model_types: List[str]
    supported_tasks: List[str]
    loaded_models: List[str]
    cache_info: Dict[str, Any]


# API 路由
@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    from datetime import datetime
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        device=str(model_manager.device),
        cuda_available=torch.cuda.is_available(),
        loaded_models=len(model_manager.cache)
    )


@app.get("/api/v1/models", response_model=ModelsResponse)
async def get_models():
    """获取支持的模型信息"""
    return ModelsResponse(
        supported_datasets=['Cora', 'CiteSeer', 'PubMed', 'ACM', 'IMDB', 'DBLP', 'syn1', 'syn2', 'syn3', 'syn4'],
        supported_model_types=['graphsage', 'dgi', 'magnn'],
        supported_tasks=['node', 'link'],
        loaded_models=list(model_manager.cache.keys()),
        cache_info=model_manager.get_cache_info()
    )


def _explain_single_node_sync(model_data: Dict[str, Any], node_id: int) -> Dict[str, Any]:
    """同步执行单节点解释的内部方法"""
    try:
        start_time = time.time()
        
        # 验证节点ID
        if node_id not in model_data['graph'].nodes():
            raise ValueError(f"节点 {node_id} 不存在于图中")
        
        # 使用explainer函数进行解释
        subgraph, importance_score = unr.explainer(
            model_data['args'],
            model_data['model'],
            model_data['graph'],
            model_data['data'],
            model_data['emb_info'],
            node_id,
            model_data['device']
        )
        
        processing_time = time.time() - start_time
        
        # 将单个重要性分数转换为列表格式以保持API一致性
        if isinstance(importance_score, (int, float)):
            importance_scores = [float(importance_score)]
        else:
            importance_scores = importance_score.tolist() if hasattr(importance_score, 'tolist') else [float(importance_score)]
        
        return {
            'node_id': node_id,
            'importance': importance_scores,
            'processing_time': processing_time,
            'subgraph_nodes': list(subgraph.nodes()) if subgraph else [node_id],
            'subgraph_edges': list(subgraph.edges()) if subgraph else []
        }
        
    except Exception as e:
        logger.error(f"节点解释失败: {e}")
        logger.error(traceback.format_exc())
        raise


@app.post("/api/v1/explain/single", response_model=ExplanationResponse)
async def explain_single_node(
    dataset: str = Query(..., description="数据集名称"),
    model: str = Query(..., description="模型名称"),
    task: str = Query(..., description="任务类型"),
    node_id: int = Query(..., description="节点ID"),
    timeout: int = Query(60, description="超时时间（秒）")
):
    """解释单个节点"""
    try:
        # 参数验证和规范化
        dataset = ModelManager.normalize_dataset(dataset)
        model = ModelManager.normalize_model(model)
        task = ModelManager.normalize_task(task)
        
        # 基本验证
        valid_datasets = ['Cora', 'CiteSeer', 'PubMed', 'ACM', 'IMDB', 'DBLP', 'syn1', 'syn2', 'syn3', 'syn4']
        valid_models = ['graphsage', 'dgi', 'magnn']
        valid_tasks = ['node', 'link']
        
        if dataset not in valid_datasets:
            raise HTTPException(status_code=400, detail=f"不支持的数据集: {dataset}")
        if model not in valid_models:
            raise HTTPException(status_code=400, detail=f"不支持的模型: {model}")
        if task not in valid_tasks:
            raise HTTPException(status_code=400, detail=f"不支持的任务: {task}")
        
        # 加载模型
        model_data, cache_info = await model_manager.load_model(dataset, model, task)
        
        # 执行解释
        result = await asyncio.wait_for(
            model_manager._run_in_thread(_explain_single_node_sync, model_data, node_id),
            timeout=timeout
        )
        
        return ExplanationResponse(
            node_id=result['node_id'],
            importance=result['importance'],
            processing_time=result['processing_time'],
            model_info={
                'dataset': dataset,
                'model': model,
                'task': task,
                'device': str(model_manager.device)
            },
            cache_info=cache_info
        )
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="请求超时")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"单节点解释失败: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


async def _generate_batch_explanations(
    model_data: Dict[str, Any],
    node_ids: List[int],
    dataset: str,
    model: str,
    task: str
):
    """生成批量解释的流式响应"""
    try:
        total_nodes = len(node_ids)
        
        # 发送开始事件
        yield f"data: {json.dumps({'type': 'start', 'total': total_nodes, 'message': '开始批量解释'})}\n\n"
        
        results = []
        for i, node_id in enumerate(node_ids):
            try:
                # 执行解释
                result = await model_manager._run_in_thread(_explain_single_node_sync, model_data, node_id)
                results.append(result)
                
                # 发送进度更新
                progress = {
                    'type': 'progress',
                    'completed': i + 1,
                    'total': total_nodes,
                    'current_node': node_id,
                    'processing_time': result['processing_time']
                }
                yield f"data: {json.dumps(progress)}\n\n"
                
            except Exception as e:
                # 发送错误信息
                error = {
                    'type': 'error',
                    'node_id': node_id,
                    'error': str(e)
                }
                yield f"data: {json.dumps(error)}\n\n"
        
        # 发送完成事件
        completion = {
            'type': 'complete',
            'results': results,
            'total_processed': len(results),
            'model_info': {
                'dataset': dataset,
                'model': model,
                'task': task,
                'device': str(model_manager.device)
            }
        }
        yield f"data: {json.dumps(completion)}\n\n"
        
    except Exception as e:
        # 发送致命错误
        error = {
            'type': 'fatal_error',
            'error': str(e)
        }
        yield f"data: {json.dumps(error)}\n\n"


@app.post("/api/v1/explain/batch")
async def explain_batch_nodes(request: BatchExplanationRequest):
    """批量解释节点（流式响应）"""
    try:
        # 加载模型
        model_data, cache_info = await model_manager.load_model(
            request.dataset, request.model, request.task
        )
        
        # 验证所有节点ID
        invalid_nodes = [nid for nid in request.node_ids 
                        if nid not in model_data['graph'].nodes()]
        if invalid_nodes:
            raise HTTPException(
                status_code=400, 
                detail=f"以下节点不存在: {invalid_nodes[:5]}{'...' if len(invalid_nodes) > 5 else ''}"
            )
        
        # 返回流式响应
        return StreamingResponse(
            _generate_batch_explanations(
                model_data, request.node_ids, 
                request.dataset, request.model, request.task
            ),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量解释失败: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """404错误处理"""
    return JSONResponse(
        status_code=404,
        content={"detail": f"路径未找到: {request.url.path}"}
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """500错误处理"""
    logger.error(f"内部服务器错误: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "内部服务器错误，请检查服务器日志"}
    )


if __name__ == "__main__":
    import uvicorn
    
    # 运行服务器
    uvicorn.run(
        "main_py38_compatible:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    ) 