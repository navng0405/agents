import httpx
import asyncio
from test_auth import create_test_token

async def test_brain_endpoint():
    token = create_test_token()
    headers = {"Authorization": f"Bearer {token}"}
    base_url = "http://127.0.0.1:8000"
    thread_id = "test_thread_123"
    

    async with httpx.AsyncClient() as client:
        msg1 = "Hey agent, change my favorite color to green , not blue"
        res1 = await client.post(f"{base_url}/chat", params={"message": msg1, "thread_id": thread_id}, headers=headers)
        print("Response to storing memory:", res1.json()['response'])

        msg2 = "What is my favorite color?"
        res2 = await client.post(f"{base_url}/chat", params={"message": msg2, "thread_id": thread_id}, headers=headers)
        print("Response to retrieving memory:", res2.json()['response'])

if __name__ == "__main__":
    asyncio.run(test_brain_endpoint())
