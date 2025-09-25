# client/ws_test.py
import asyncio
import json
import websockets

async def main():
    uri = "ws://127.0.0.1:8000/stream"
    async with websockets.connect(uri) as ws:
        print("Connected. Listening for events...")
        # gửi ping định kỳ để giữ kết nối (phù hợp với handler hiện tại)
        async def pinger():
            while True:
                try:
                    await ws.send("ping")
                except:
                    break
                await asyncio.sleep(2)

        asyncio.create_task(pinger())

        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            print("EVENT:", data["id"], data.get("person"))

asyncio.run(main())