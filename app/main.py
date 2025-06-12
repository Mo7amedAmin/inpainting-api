from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from PIL import Image
import io
from app.pipeline import process_image

app = FastAPI()

@app.post("/generate/")
async def generate(prompt: str, image: UploadFile = File(...)):
    # اقرأ الصورة المرفوعة
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

    # عالج الصورة باستخدام البايبلاين
    result = process_image(pil_image, prompt)

    # احفظ الصورة الناتجة مؤقتًا
    result_path = "result.png"
    result.save(result_path)

    # رجع الصورة كـ Response
    return FileResponse(result_path, media_type="image/png")
