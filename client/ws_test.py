import asyncio, json, websockets

async def main():
    uri = "ws://127.0.0.1:8000/stream"
    async with websockets.connect(uri) as ws:
        print("Connected. Listening for events...")
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=10)
                data = json.loads(msg)
                print("EVENT:", data["id"], data.get("person"))
            except asyncio.TimeoutError:
                print("No event received in 10s, still alive...")

if __name__ == "__main__":
    asyncio.run(main())