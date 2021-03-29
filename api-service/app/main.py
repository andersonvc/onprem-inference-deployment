import os
from typing import Optional
import httpx
import asyncio

from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.responses import JSONResponse,RedirectResponse

app = FastAPI()

ts_ip = os.getenv('TORCHSERVE_IP','192.168.1.109')
mgmt_port = os.getenv('TORCHSERVE_MANAGEMENT_PORT','8081')
inference_port = os.getenv('TORCHSERVE_INFERENCE_PORT','8080')

async def get_request(url):
    async with httpx.AsyncClient() as client:
        return await client.get(url)

async def post_request(url,data=None):
    async with httpx.AsyncClient() as client:
        return await client.post(url,data=data) if data else client.post(url,data={})



@app.get("/")
async def redirect():
    response = RedirectResponse(url='/docs')
    return response

@app.get('/models')
async def get_models():
    url = f'http://{ts_ip}:{mgmt_port}/models'
    resp = await get_request(url)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code,detail=f'Error: failed to call /models endpoint\n{resp.json()}')
    return resp.json()

@app.post("/classify-image")
async def classify_image(file: bytes = File(...)):
    model_name='densenet161'
    url=f'http://{ts_ip}:{inference_port}/predictions/{model_name}'
    resp = await post_request(url,file)
    return resp.json()

@app.post("/object-detection")
async def classify_image(file: bytes = File(...)):
    model_name='fastrcnn'
    url=f'http://{ts_ip}:{inference_port}/predictions/{model_name}'
    resp = await post_request(url,file)
    return resp.json()

@app.post("/set-model-version")
async def set_model_version(model_name,model_version):
    url = f'http://{ts_ip}:{mgmt_port}/models/{model_name}/{model_version}'
    resp = await get_request(url)
    #if resp.status_code != 200:
    #    return Response(status_code=resp.status_code,content=resp.content)
    
    
    url = f'http://{ts_ip}:{mgmt_port}/models/{model_name}/{model_version}/set-default'
    resp = await get_request(url)
    return resp.json()

@app.post("/upload-model")
async def upload_model(upload_file:UploadFile = File(...)):
    file_location = f"/torchdata-model-store/{upload_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(upload_file.file.read())
    
    is_synchronous='false'
    initial_workers = 1

    url = f'http://{ts_ip}:{mgmt_port}/models?url={upload_file.filename}&initial_workers={str(initial_workers)}&synchronous={is_synchronous}'

    data = {
        'url':upload_file.filename,
        'synchronous':is_synchronous,
        'initial_workers':str(initial_workers),
    }

    resp = await post_request(url,data)
    if resp.status_code != 200:
        return Response(status_code=resp.status_code,content=resp.content)
    
    return resp.json()

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}
