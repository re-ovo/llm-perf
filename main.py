import yaml
import asyncio
import time
from typing import Dict, Any, List, Tuple
import aiohttp
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TaskID
from openai import AsyncOpenAI
import json
import statistics

async def measure_performance(provider: Dict[str, Any], 
                            model: str,
                            prompt: str) -> tuple[float, float]:
    """测量特定模型的TTFT和TPS"""
    
    # 创建OpenAI客户端
    client = AsyncOpenAI(
        api_key=provider['api_key'],
        base_url=provider['base_url']
    )
    
    start_time = time.time()
    first_token_time = None
    total_tokens = 0
    
    # 使用OpenAI SDK的流式响应
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True}
    )
    
    async for chunk in stream:
        # 检查是否有任何类型的输出
        if chunk.choices and chunk.choices[0].delta:
            if first_token_time is None:
                first_token_time = time.time()
        
        # 如果有usage信息，获取token数量
        if hasattr(chunk, 'usage') and chunk.usage:
            total_tokens = chunk.usage.completion_tokens
    
    end_time = time.time()
    
    ttft = first_token_time - start_time if first_token_time else 0
    total_time = end_time - start_time
    tps = total_tokens / total_time if total_time > 0 and total_tokens > 0 else 0
    
    return ttft, tps

async def measure_performance_multiple_times(
    provider: Dict[str, Any], 
    model: str,
    prompt: str,
    num_runs: int = 3
) -> tuple[float, float]:
    """多次测量特定模型的性能并返回平均值"""
    ttfts = []
    tpss = []
    
    for _ in range(num_runs):
        ttft, tps = await measure_performance(provider, model, prompt)
        ttfts.append(ttft)
        tpss.append(tps)
    
    avg_ttft = statistics.mean(ttfts)
    avg_tps = statistics.mean(tpss)
    
    return avg_ttft, avg_tps

async def measure_performance_with_progress(
    provider: Dict[str, Any], 
    model: str,
    progress: Progress,
    task_id: TaskID,
    prompt: str,
    num_runs: int = 3
) -> tuple[float, float]:
    """测量特定模型的TTFT和TPS，并更新进度"""
    try:
        result = await measure_performance_multiple_times(provider, model, prompt, num_runs)
        progress.update(task_id, advance=1)
        return result
    except Exception as e:
        progress.update(task_id, advance=1)
        raise e

async def main():
    # 读取配置文件
    with open("config.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    prompt = config.get("prompt", "")
    providers = config.get("providers", {})
    num_runs = config.get("num_runs", 3)  # 从配置文件中获取运行次数，默认为3次
    console = Console()
    
    # 创建结果表格
    table = Table(title=f"LLM提供商性能评估 (每个模型测试{num_runs}次的平均值)")
    table.add_column("提供商", style="cyan")
    table.add_column("模型", style="magenta")
    table.add_column("平均TTFT (秒)", justify="right", style="green")
    table.add_column("平均TPS (tokens/s)", justify="right", style="yellow")
    
    # 创建任务列表
    tasks = []
    task_info = []
    
    # 计算总任务数
    total_tasks = sum(len(provider_info.get("models", [])) for provider_id, provider_info in providers.items())
    
    # 创建进度条
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        main_task = progress.add_task("[cyan]评估LLM性能中...", total=total_tasks)
        
        for provider_id, provider_info in providers.items():
            for model in provider_info.get("models", []):
                task = measure_performance_with_progress(
                    provider_info, 
                    model, 
                    progress, 
                    main_task,
                    prompt,
                    num_runs
                )
                tasks.append(task)
                task_info.append((provider_info["name"], model))
        
        # 并行执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 处理结果
    for i, result in enumerate(results):
        provider_name, model = task_info[i]
        if isinstance(result, Exception):
            table.add_row(
                provider_name,
                model,
                "错误",
                str(result)
            )
        else:
            ttft, tps = result
            table.add_row(
                provider_name,
                model,
                f"{ttft:.3f}",
                f"{tps:.2f}"
            )
    
    console.print(table)

if __name__ == "__main__":
    asyncio.run(main())
