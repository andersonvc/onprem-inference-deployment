import os
from typing import Optional
import httpx
import asyncio

from fastapi import FastAPI, File, UploadFile

app = FastAPI()

ts_ip = '192.168.1.109' #os.getenv('TORCHSERVE_IP','192.168.1.109')
ts_port = '8081' #os.getenv('TORCHSERVE_PORT','8081')


async def get_request(url):
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
    return resp.json()

async def post_request(url,body):
    async with httpx.AsyncClient() as client:
        resp = await client.post(url,data=body)
    return resp.json()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/models')
async def get_models():
    return await get_request(f'http://{ts_ip}:{ts_port}/models')

@app.post("/classify-image")
async def classify_image(file: bytes = File(...)):
    model_name='densenet161'
    #model_version='1'
    url=f'http://{ts_ip}:8080/predictions/{model_name}'

    print(url)
    
    return await post_request(url,file)


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}
