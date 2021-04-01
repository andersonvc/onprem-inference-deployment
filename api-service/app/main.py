import os
from typing import Optional
import httpx
import asyncio

from fastapi import FastAPI, File, UploadFile, HTTPException, Response, WebSocket
from fastapi.responses import JSONResponse,RedirectResponse
from starlette.responses import FileResponse, HTMLResponse

from fastapi.staticfiles import StaticFiles

app_title="Model Serving: External API Interface"
tags_metadata = [
    {"name": "Management", "description": "API for managing prod model deployments"},
    {"name": "Models", "description": "Deployed model inference endpoint"},
    {"name": "Streaming", "description": "Streaming video websocket endpoint"},
]

app = FastAPI(
    title=app_title,
    openapi_tags=tags_metadata)

app.mount("/public", StaticFiles(directory="public"), name="public")


ts_ip = os.getenv('TORCHSERVE_IP','192.168.1.109')
mgmt_port = os.getenv('TORCHSERVE_MANAGEMENT_PORT','8081')
inference_port = os.getenv('TORCHSERVE_INFERENCE_PORT','8080')

async def get_request(url):
    async with httpx.AsyncClient() as client:
        return await client.get(url)

async def post_request(url,data=None):
    async with httpx.AsyncClient() as client:
        return await client.post(url,data=data) if data else client.post(url,data={})


"""
FIXME: Clean up this section
"""

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def publisher_connect(self,websocket: WebSocket):
        await websocket.accept()

    async def connect(self,websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self,websocket:WebSocket):
        self.active_connections.remove(websocket)

    async def receipt(self,websocket: WebSocket,message:str):
        await websocket.send_text(f'received {message}')

    async def broadcast(self,message:str):
        for conn in self.active_connections:
            await conn.send_text(message)

manager = ConnectionManager()


@app.websocket("/ws-publish/{client_id}")
async def websocket_publish(websocket: WebSocket):
    await manager.publisher_connect(websocket)
    try:
        while True:

            data = await websocket.receive_bytes()

            data = np.fromstring(data, dtype=np.uint8)
            raw_image = cv2.imdecode(data, cv2.IMREAD_COLOR).reshape((720,1080,3))

            await manager.receipt(websocket,'1')

            det = inference_detector(model,raw_image)

            person_center = det[0][det[0][:,-1]>.25]
            pc_y,pc_x = (person_center[:,2]+person_center[:,0])//2,(person_center[:,3]+person_center[:,1])//2


            processed_image = model.show_result(raw_image,det,score_thr=threshold,show=False)

            for y,x in zip(pc_y,pc_x):
                cv2.circle(processed_image,(y,x),7,(255,255,255),-1)
                cv2.circle(processed_image,(y,x),5,(0,0,255),-1)


            processed_image = cv2.imencode('.jpg',processed_image)[1]

            output_stream = base64.b64encode(processed_image)
            #print(len(output_stream))

            await manager.broadcast(output_stream)

    except WebSocketDisconnect:
        #manager.disconnect(websocket)
        await manager.broadcast(f"Camera Disconnected")


"""
FIXME End Cleanup section
"""




@app.get("/", tags=["Streaming"])
async def redirect():
    response = RedirectResponse(url='/wstest')
    return response

@app.get('/models',tags=["Models"])
async def get_models():
    url = f'http://{ts_ip}:{mgmt_port}/models'
    resp = await get_request(url)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code,detail=f'Error: failed to call /models endpoint\n{resp.json()}')
    return resp.json()

@app.post("/classify-image",tags=["Models"])
async def classify_image(file: bytes = File(...)):
    model_name='densenet161'
    url=f'http://{ts_ip}:{inference_port}/predictions/{model_name}'
    resp = await post_request(url,file)
    return resp.json()

@app.post("/object-detection",tags=["Models"])
async def classify_image(file: bytes = File(...)):
    model_name='fastrcnn'
    url=f'http://{ts_ip}:{inference_port}/predictions/{model_name}'
    resp = await post_request(url,file)
    return resp.json()

@app.post("/set-model-version",tags=["Management"])
async def set_model_version(model_name,model_version):
    url = f'http://{ts_ip}:{mgmt_port}/models/{model_name}/{model_version}'
    resp = await get_request(url)
    #if resp.status_code != 200:
    #    return Response(status_code=resp.status_code,content=resp.content)
    
    url = f'http://{ts_ip}:{mgmt_port}/models/{model_name}/{model_version}/set-default'
    resp = await get_request(url)
    return resp.json()

@app.post("/upload-model",tags=["Management"])
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


@app.get("/wstest", tags=["Streaming"])
def read_index():
    #return FileResponse('public/html/wstext.html')
    return FileResponse('public/simple_webrtc/index.html')

"""
Websocket Connections
"""

@app.websocket('/ws')
async def ws(websocket:WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message was received: {data}")
