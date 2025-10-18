from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from typing import Optional
import numpy as np
import io
import torch


from predetect import count_damage_instances
from predict2 import analyze_damage_parts


app = FastAPI()

# (ตัวเลือก) เปิด CORS ให้ frontend เรียกได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://cdd-project.vercel.app"],  # ระบุโดเมนของคุณแทน "*" ในโปรดักชัน
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}



# ---------- ใหม่: วิเคราะห์ damage -> part ด้วย mask IoU ----------
@app.post("/detect/analyze")
async def detect_analyze(
    file: UploadFile = File(...),
    conf_parts: float = 0.15,
    conf_damage: float = 0.25,
    imgsz: int = 640,
    mask_iou_thresh: float = 0.08,
    render_overlay: bool = False,  # ถ้าจริง จะได้ base64 ของภาพซ้อนผลลัพธ์
):
    """
    วิเคราะห์ภาพ:
      - รัน parts model และ damage model
      - จับคู่ด้วย Mask IoU
      - ส่งรายการ parts ที่พบความเสียหายกลับ
      - ถ้า render_overlay=True จะได้ภาพซ้อนเป็น base64 ในฟิลด์ overlay_image_b64
    """
    image_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_image = np.array(pil_image)

    with torch.inference_mode():
        out = analyze_damage_parts(
            np_image,
            conf_parts=conf_parts,
            conf_damage=conf_damage,
            imgsz=imgsz,
            mask_iou_thresh=mask_iou_thresh,
            render_overlay=render_overlay,
        )
    return JSONResponse(content=out)