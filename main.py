from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from PIL import Image
import io
from app.pipeline import process_image

app = FastAPI()

@app.post("/generate/")
async def generate(prompt: str, image: UploadFile = File(...)):
    
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

    result = process_image(pil_image, prompt)

    result_path = "result.png"
    result.save(result_path)

    return FileResponse(result_path, media_type="image/png")
