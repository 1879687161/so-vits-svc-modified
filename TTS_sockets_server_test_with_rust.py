import psutil
import subprocess
import time
from fastapi import FastAPI, WebSocket
import asyncio
import websockets
from TTS_inference_V2 import TTS
import traceback
import json
import aioredis
import os

# 定义项目根目录
#project_root = "/home/tyb0v0/Rustd/run_tts/so-vits-svc-modified"

# 更改当前工作目录
#os.chdir(project_root)

app = FastAPI()
redis_url = "redis://127.0.0.1/"
#llm = LLM()

args = {
    "model_path": "logs/44k/G_36800.pth",
    "config_path": "logs/44k/config.json",
    "clip": 0,
    "clean_names": ["audio.wav"],
    "trans": [0],
    "spk_list": ['rjdy'],
    "auto_predict_f0": False,
    "cluster_model_path": "",
    "cluster_infer_ratio": 0,
    "linear_gradient": 0,
    "f0_predictor": "rmvpe",
    "enhance": False,
    "shallow_diffusion": False,
    "use_spk_mix": False,
    "loudness_envelope_adjustment": 1,
    "feature_retrieval": False,
    "diffusion_model_path": "logs/44k/diffusion/model_0.pt",
    "diffusion_config_path": "logs/44k/diffusion/config.yaml",
    "k_step": 100,
    "second_encoding": False,
    "only_diffusion": False,
    "slice_db": -40,
    "device": None,
    "noice_scale": 0.4,
    "pad_seconds": 0.5,
    "wav_format": 'wav',
    "linear_gradient_retain": 0.75,
    "enhancer_adaptive_key": 0,
    "f0_filter_threshold": 0.05,
    "edge_tts_voice": 'zh-CN-YunxiNeural',
    "edge_tts_output_path": 'C:/Users/zhangkaihao/so-vits-svc-modified/raw/audio.wav',
    "edge_tts_rate": 0,
    "edge_tts_volume": 0,
    "edge_tts_pitch": '0%',
    "edge_tts_text": '请输入您的文本'
}
M = TTS(args)
M.process_args()

async def process_queue(redis_url: str) -> None:
    redis = await aioredis.from_url(redis_url, encoding='utf-8')

    while True:
        _, task_string = await redis.blpop("my_queue")

        if task_string:
            task_data = json.loads(task_string)

            # 提取任务数据并处理
            input_text = task_data["input"]

            try:
                wav_bin_data = M.final_API(input_text)

                # 任务完成后，将任务完成的消息发送到 Redis 队列
                completed_task = {
                    #"id": task_data["id"], 
                    "status": "completed",
                    "result": "Audio data generated successfully",
                }
                await redis.rpush("completed_tasks", json.dumps(completed_task)) # 任务完成后，将任务完成的消息发送到 Redis 队列

            except Exception as e:
                print(f"An error occurred during processing: {e}")
                print(traceback.format_exc())

                # 将任务标记为错误并将其添加到 Redis 队列
                error_task = {
                    #"id": task_data["id"],
                    "status": "error",
                    "result": "Error: unable to process the request",
                }
                await redis.rpush("completed_tasks", json.dumps(error_task))

        # 等待一会儿再次检查队列
        await asyncio.sleep(0.1)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await handle_request(websocket)

if __name__ == "__main__":
    import uvicorn

    asyncio.get_event_loop().run_until_complete(process_queue(redis_url))
    uvicorn.run(app, host="127.0.0.1", port=8000)