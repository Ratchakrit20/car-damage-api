from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import numpy as np
import cv2
import io
import torch
from predict2 import analyze_damage_parts

app = FastAPI()

# เปิด CORS ให้ frontend เรียกได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://cdd-project.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/detect/analyze")
async def detect_analyze(
    file: UploadFile = File(...),
    conf_parts: float = 0.15,
    conf_damage: float = 0.25,
    imgsz: int = 640,
    mask_iou_thresh: float = 0.08,
    render_overlay: bool = False,
    preprocess: bool = Query(False, description="เลือกว่าจะ Preprocess ภาพก่อนหรือไม่"),
):
    """
    วิเคราะห์ภาพ:
      - เลือกได้ว่าจะใช้ภาพแบบเดิม หรือแบบที่ผ่าน preprocessing
      - รัน parts model และ damage model
      - จับคู่ด้วย Mask IoU
    """
    image_bytes = await file.read()

    # --- ✅ เลือกโหมดการเตรียมภาพ ---
    if preprocess:
        np_image = preprocess_image(image_bytes, target_size=imgsz)
        mode = "preprocessed"
    else:
        np_image = resize_image(image_bytes, target_size=imgsz)
        mode = "original"

    with torch.inference_mode():
        out = analyze_damage_parts(
            np_image,
            conf_parts=conf_parts,
            conf_damage=conf_damage,
            imgsz=imgsz,
            mask_iou_thresh=mask_iou_thresh,
            render_overlay=render_overlay,
            preprocess=preprocess,  
        )

    # ✅ เพิ่มข้อมูล mode ในผลลัพธ์เพื่อให้ฝั่ง frontend แยกได้
    out["processing_mode"] = mode

    return JSONResponse(content=out)

def preprocess_image(image_bytes, target_size=640):
    """ปรับภาพก่อนทำนาย (ใช้ CLAHE บน LAB เพื่อเพิ่ม contrast สีและแสง)"""
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((target_size, target_size))

    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 🔸 แปลงเป็น LAB เพื่อปรับความสว่างในช่อง L
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 🔸 ใช้ CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge((l_eq, a, b))
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # 🔸 แปลงกลับเป็น RGB NumPy
    img_ready = cv2.cvtColor(img_eq, cv2.COLOR_BGR2RGB)
    return img_ready

def resize_image(image_bytes, target_size=640):
    """ปรับภาพก่อนทำนาย"""
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)  # Auto-Orient
    image = image.convert("RGB")
    image = image.resize((target_size, target_size))

    # แปลงจาก PIL → NumPy (RGB)
    img_ready = np.array(image)
    return img_ready