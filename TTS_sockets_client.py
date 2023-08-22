import asyncio
import websockets
import time
from starlette.websockets import WebSocketDisconnect
from websockets.exceptions import ConnectionClosedOK

""" async def websocket_client():
    async with websockets.connect('ws://127.0.0.1:8086/ws') as websocket:
        while True:
            # 获取用户输入
            message = input("请输入要发送的文本：")
            # 发送消息
            await websocket.send(message)

            # 接收服务器响应
            wav_bin_data = await websocket.recv()
            #print(wav_bin_data)
            print(f"Wav data received!")

            if message.lower() == "exit":
                print("退出客户端。")
                break """

        #time.sleep(1)
        #print("===")

        #await websocket.send(message)
        #wav_bin_data = await websocket.recv()
        #print(wav_bin_data)
        # 遍历字节并将其转换为二进制字符串
        #print(f"Wav data received!")

#asyncio.get_event_loop().run_until_complete(websocket_client())
#asyncio.get_event_loop().run_forever()

async def heartbeat(websocket):
    while True:
        await asyncio.sleep(30)  # 每30秒发送一次心跳
        await websocket.send("heartbeat")
        response = await websocket.recv()

        if response == "heartbeat_ack":
            print("心跳确认已收到")
        else:
            print(f"意外响应：{response}")


async def handle_user_input(websocket,message_queue):
    while True:
        try:
            # 获取用户输入
            message = input("请输入要发送的文本：")

            # 保存消息到队列
            message_queue.append(message)

            # 发送消息
            await websocket.send(message)

            # 接收服务器响应
            wav_bin_data = await websocket.recv()
            #print(wav_bin_data)
            print(f"Wav data received!")

            # 如果发送成功，从队列中删除消息
            message_queue.pop(0)

            if message.lower() == "exit":
                print("退出客户端。")
                break
    
        except ConnectionClosedOK:
            print("连接已正常关闭")
            break

        except websockets.exceptions.ConnectionClosed:
            print("连接异常关闭")
            break
    
    #return message_queue


async def continue_user_input(websocket,message_queue):
    while True:
        # 发送消息
        await websocket.send(message_queue[0])

        # 接收服务器响应
        wav_bin_data = await websocket.recv()

        #print(wav_bin_data)
        print(f"Wav data received!")
        break


async def websocket_client():
     message_queue = []  # 声明用于存储消息的列表
     while True:
        try:
            async with websockets.connect('ws://10.1.12.199:7076/ws') as websocket:
                # 创建并运行处理用户输入和心跳的协程
                user_input_task = asyncio.ensure_future(handle_user_input(websocket, message_queue))
                #heartbeat_task = asyncio.ensure_future(heartbeat(websocket))
            
                # 等待任务完成
                #await asyncio.gather(user_input_task, heartbeat_task)
                await asyncio.gather(user_input_task)

                # 获取返回的 input_queue
                returned_queue = user_input_task.result()

        except websockets.exceptions.ConnectionClosed:
            print("连接已断开，尝试重新连接")
            await asyncio.sleep(3)  # 等待3秒后尝试重新连接
            #if message_queue:
                #user_input_task_countinue = asyncio.ensure_future(continue_user_input(websocket, returned_queue))
                # 等待任务完成
                #await asyncio.gather(user_input_task_countinue)
                #message_queue.pop(0)

asyncio.get_event_loop().run_until_complete(websocket_client())

#loop = asyncio.get_event_loop()
#loop.run_until_complete(websocket_client())
#loop.create_task(heartbeat())
#loop.run_forever()