from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from PIL import Image
import io
import uuid
import os
from fastapi.responses import StreamingResponse
from fastapi import Form
from fastapi.middleware.cors import CORSMiddleware

from app.pipeline import process_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate/")
async def generate(prompt: str = Form(...), image: UploadFile = File(...)):
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

    result = process_image(pil_image, prompt)

    img_bytes = io.BytesIO()
    result.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    return StreamingResponse(img_bytes, media_type="image/png")