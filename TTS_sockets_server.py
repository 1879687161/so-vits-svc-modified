import psutil
import subprocess
import time
from fastapi import FastAPI, WebSocket
import asyncio
import websockets
from TTS_inference_V2 import TTS
import traceback
#from TTS_inference_V2 import parse_arguments

app = FastAPI()
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
    "f0_predictor": "harvest",
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

""" @app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received: {data}")
            #response, querry_result, debug_info = llm.predict(data)
            wav_bin_data = M.final_API(data)
            print("Audio output finished!")
            await websocket.send_bytes(wav_bin_data)
            print("Audio data sent!")
            #pass
    except:
        # Disconnect
        await websocket.close()
    
    ######### Debug用的,一般情况不用。#########
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received: {data}")

            try:
                wav_bin_data = M.final_API(data)
                print("Audio output finished!")
                await websocket.send_bytes(wav_bin_data)
                print("Audio data sent!")
            except Exception as e:
                print(f"An error occurred during processing: {e}")
                print(traceback.format_exc())
                await websocket.send("Error: unable to process the request")
    except Exception as e:
        print(f"An error occurred during the connection: {e}")
        print(traceback.format_exc()) """


async def handle_request(websocket: WebSocket):
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received: {data}")
            
            if data == "heartbeat":
                await websocket.send_text("heartbeat_ack")
                print(f"心跳消息已收到并确认: {data}")
            
            else:
                try:
                    wav_bin_data = M.final_API(data)
                    print("Audio output finished!")
                    await websocket.send_bytes(wav_bin_data)
                    print("Audio data sent!")
                except Exception as e:
                    print(f"An error occurred during processing: {e}")
                    print(traceback.format_exc())
                    await websocket.send("Error: unable to process the request")
    except Exception as e:
        print(f"An error occurred during the connection: {e}")
        print(traceback.format_exc())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await handle_request(websocket)