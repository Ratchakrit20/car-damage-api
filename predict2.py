from ultralytics import YOLO
import numpy as np
import cv2
import torch
import base64
from typing import Dict, Any, List

# ---------- Utilities ----------
def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(inter) / float(union) if union else 0.0

def mask_center(mask: np.ndarray) -> np.ndarray:
    """centroid ของ mask แบบ (y, x) ถ้า mask ว่างให้คืนค่า (inf, inf) เพื่อไม่ให้จับคู่ผิด"""
    ysx = np.argwhere(mask)
    if ysx.size == 0:
        return np.array([np.inf, np.inf], dtype=np.float32)
    return ysx.mean(axis=0).astype(np.float32)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_PARTS  = YOLO("model/parts/best.pt").to(_DEVICE)
_MODEL_DAMAGE = YOLO("model/damage/best.pt").to(_DEVICE)
_PARTS_NAMES  = _MODEL_PARTS.names
_DAMAGE_NAMES = _MODEL_DAMAGE.names


def analyze_damage_parts(
    np_image: np.ndarray,
    conf_parts: float = 0.25,
    conf_damage: float = 0.25,
    imgsz: int = 640,
    mask_iou_thresh: float = 0.03,    # เกณฑ์ IoU ระหว่าง part ↔ damage
    inside_thresh: float = 0.10,      # เกณฑ์สัดส่วน damage ที่อยู่ “ใน” part
    scratch_dilate_iter: int = 1,     # ขยายรอยขีดก่อนจับคู่ (ช่วยจับคู่กับ part ได้ดีขึ้น)
    cluster_iou_sameclass: float = 0.10,   # รวมความเสียหายชนิดเดียวกันถ้า IoU > ค่านี้
    cluster_dist_sameclass: float = 120.0, # หรือระยะ centroid น้อยกว่าค่านี้ (พิกเซล)
    prox_iou_unmatched: float = 0.05,      # รวม unmatched damages ถ้า IoU > ค่านี้
    prox_dx_unmatched: float = 200.0,      # หรือ |dx| < ค่านี้
    prox_dy_unmatched: float = 250.0,      # และ |dy| < ค่านี้
    max_classes_per_part: int = 0,         # 0=ส่งทุกชนิด, 1=ส่งเฉพาะชนิดที่เด่นสุด
    render_overlay: bool = False
) -> Dict[str, Any]:

    """
    วิเคราะห์ภาพรถ:
      - จับคู่ damage ↔ part ด้วย IoU/Inside
      - รวมความเสียหายชนิดเดียวกันที่อยู่ใกล้/ทับกันใน part เดียวกัน
      - คิด coverage ของแต่ละคลาสและ coverage รวมทุกคลาสต่อชิ้นส่วน
      - สร้าง Virtual Part สำหรับ damage ที่ไม่ทับ part ใดเลย (cluster ด้วย IoU/ระยะ)

    return:
      {
        ok, width, height,
        parts: [
          {
            part, bbox,
            damages: [
              {class, confidence, count, mask_iou, mask_coverage}
            ],
            damage_coverage, damage_coverage_percent,
            is_virtual?   # มีเฉพาะชิ้นส่วนจำลอง
          }, ...
        ],
        overlay_image_b64?, overlay_mime?
      }
    """
    

    H, W = np_image.shape[:2]

    # ---------- Make input identical to your test pipeline ----------
    # 1) dtype → uint8 [0..255]
    if np_image.dtype != np.uint8:
        np_image = (np.clip(np_image, 0, 1) * 255).astype(np.uint8)

    # 2) สีภาพต้องเป็น RGB (เพราะภาพมาจาก PIL อยู่แล้วเป็น RGB)
    #   ถ้าใครเรียกฟังก์ชันนี้ด้วยภาพ BGR (เช่นจาก cv2) ให้ตรวจและสลับเฉพาะกรณีจำเป็น
    if np_image.ndim == 3 and np_image.shape[2] == 3:
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)


    np_image = np.ascontiguousarray(np_image)  # กัน edge-case ของ OpenCV/Ultralytics
    # บันทึกรูปเพื่อเทียบกับตอนเทสโมเดล
    cv2.imwrite("debug_before_model.jpg", cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR))
    print("✅ Saved debug_before_model.jpg (exactly the same input sent to YOLO)")
    print(_MODEL_PARTS.device, _MODEL_DAMAGE.device)
    print(torch.cuda.is_available())

    # ---------- 0) รันโมเดล ----------
    with torch.inference_mode():
        parts_result  = _MODEL_PARTS.predict(
            source=np_image,
            imgsz=imgsz,
            conf=conf_parts,
            iou=0.5,
        )[0]
        damage_result = _MODEL_DAMAGE.predict(
            source=np_image,
            imgsz=imgsz,
            conf=conf_damage,   # ลอง 0.15 ถ้าของ test ใช้ 0.15
        )[0]

    # damage_result.show()  # ดูผลคร่าว ๆ
    # parts_result.show()   # ดูผลคร่าว ๆ
    print("Original shape used by YOLO:", parts_result.orig_shape)
    damage_result.save(filename="output_api.jpg")  # บันทึกผลลัพธ์
    if parts_result.masks is None or damage_result.masks is None:
        return {"ok": True, "width": W, "height": H, "parts": [], "message": "no masks from one of the models"}

    # ---------- 1) อัปสเกล mask เป็นขนาดภาพจริง ----------
    def upsample_bool(m: np.ndarray) -> np.ndarray:
        m8 = (m > 0.5).astype(np.uint8)
        return cv2.resize(m8, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

    part_masks   = np.stack([upsample_bool(m) for m in parts_result.masks.data.cpu().numpy()], axis=0)
    damage_masks = np.stack([upsample_bool(m) for m in damage_result.masks.data.cpu().numpy()], axis=0)

    overlay_bgr = np_image[:, :, ::-1].copy() if render_overlay else None  # RGB->BGR
    out_parts: List[Dict[str, Any]] = []

    # จะเก็บดัชนี damage ที่ "จับคู่แล้ว" ระดับอินสแตนซ์ (ไม่ใช่ตามชื่อคลาส)
    matched_damage_indices: set[int] = set()

    # ---------- 2) เดินทีละชิ้นส่วน ----------
    for i, part_box in enumerate(parts_result.boxes):
        if i >= part_masks.shape[0]:
            continue

        part_cls  = int(part_box.cls.item())
        part_name = _PARTS_NAMES[part_cls]
        x1, y1, x2, y2 = map(int, part_box.xyxy[0].tolist())
        part_mask = part_masks[i]
        part_area = float(part_mask.sum()) or 1.0  # กันหารศูนย์

        # per_class: { "dent": [cluster1, cluster2, ...], "scratch": [...] }
        # cluster = {"union": bool mask (ของคลัสเตอร์), "conf_max": float, "count": int, "center": (y,x)}
        per_class: Dict[str, List[Dict[str, Any]]] = {}

        # ---------- 3) เดินทุก damage แล้วตัดสินว่า “ทับ part พอไหม” ----------
        for j, damage_box in enumerate(damage_result.boxes):
            if j >= damage_masks.shape[0]:
                break

            damage_cls  = int(damage_box.cls.item())
            damage_name = _DAMAGE_NAMES[damage_cls]
            confidence  = float(damage_box.conf.item())

            dmask_real  = damage_masks[j]      # mask จริง ใช้คำนวณพื้นที่
            dmask_match = dmask_real

            # ขยายเฉพาะตอนจับคู่ (ช่วยกรณีรอยขีดบาง ๆ)
            if damage_name.lower() in ("scratch", "scratches") and scratch_dilate_iter > 0:
                kernel = np.ones((3, 3), np.uint8)
                dmask_match = cv2.dilate(dmask_real.astype(np.uint8), kernel, scratch_dilate_iter).astype(bool)

            inter  = np.logical_and(part_mask, dmask_match).sum()
            if inter == 0:
                continue

            union  = np.logical_or(part_mask, dmask_match).sum()
            iou    = float(inter) / float(union) if union else 0.0
            inside = float(inter) / float(dmask_match.sum()) if dmask_match.sum() else 0.0

            # ผ่านถ้าอย่างน้อยหนึ่งในสองเกณฑ์ถึง threshold
            if (iou < mask_iou_thresh) and (inside < inside_thresh):
                continue

            # ---- 4) รวมเป็น "คลัสเตอร์" เฉพาะชนิดเดียวกันที่อยู่ใกล้/ทับกันใน part นี้ ----
            # ถ้าอยู่ไกลกันมากจะไม่รวม จะกลายเป็นอีกอินสแตนซ์หนึ่งของคลาสเดียวกัน
            d_center = mask_center(dmask_real)
            merged = False
            for cluster in per_class.get(damage_name, []):
                iou_prev = mask_iou(cluster["union"], dmask_real)
                c_center = cluster["center"]
                dist = float(np.hypot(*(d_center - c_center)))  # พิกเซล

                if (iou_prev >= cluster_iou_sameclass) or (dist <= cluster_dist_sameclass):
                    cluster["union"] |= dmask_real
                    cluster["conf_max"] = max(cluster["conf_max"], confidence)
                    cluster["count"] += 1
                    cluster["center"] = mask_center(cluster["union"])
                    merged = True
                    break

            if not merged:
                per_class.setdefault(damage_name, []).append({
                    "union": dmask_real.copy(),
                    "conf_max": confidence,
                    "count": 1,
                    "center": d_center
                })

            # ทำเครื่องหมายว่า damage index นี้ถูกจับคู่กับ part แล้ว
            matched_damage_indices.add(j)

        # ไม่มี damage ที่ผ่านเกณฑ์สำหรับ part นี้
        if not per_class:
            continue

        # ---------- 5) คิด coverage ต่อคลาส + คิด "coverage รวมทุกคลาส" ----------
        damages: List[Dict[str, Any]] = []
        damage_union_all = np.zeros_like(part_mask, dtype=bool)

        for cls_name, clusters in per_class.items():
            for info in clusters:
                umask = info["union"]
                damage_union_all |= umask

                inter = np.logical_and(part_mask, umask).sum()
                cover = float(inter) / part_area  # สัดส่วนพื้นที่ของคลัสเตอร์นี้ต่อพื้นที่ชิ้นส่วน
                union_mu = np.logical_or(part_mask, umask).sum()
                miou = float(inter) / float(union_mu) if union_mu else 0.0

                damages.append({
                    "class": cls_name,
                    "confidence": round(info["conf_max"], 4),
                    "count": info["count"],                 # อินสแตนซ์ที่ถูกรวมในคลัสเตอร์นี้
                    "mask_iou": round(miou, 4),             # IoU (คลัสเตอร์นี้ ↔ part)
                    "mask_coverage": round(cover, 4),       # สัดส่วนพื้นที่คลัสเตอร์นี้/ชิ้นส่วน
                })

        # coverage รวมทุกคลาส (union cross-class)
        total_cover = float(np.logical_and(part_mask, damage_union_all).sum()) / part_area

        # ต้องการส่งเฉพาะ “ตัวเด่นสุด” ต่อ part
        if max_classes_per_part == 1 and damages:
            damages.sort(key=lambda d: (d["mask_coverage"], d["confidence"], d["count"]), reverse=True)
            damages = damages[:1]

        # ---------- 6) วาด overlay ----------
        if render_overlay and overlay_bgr is not None:
            # ระบาย union ของทุกคลาสให้ดูภาพรวมความเสียหายของ part
            mask_uint8 = (damage_union_all.astype(np.uint8) * 255)
            colored = np.zeros_like(overlay_bgr, dtype=np.uint8)
            color = np.array([50, 180, 255], dtype=np.uint8)  # ฟ้าอ่อน (ไม่ fix ก็ได้)
            for c in range(3):
                colored[:, :, c] = (mask_uint8 * int(color[c])) // 255
            overlay_bgr = cv2.addWeighted(overlay_bgr, 1.0, colored, 0.35, 0.0)
            cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(overlay_bgr, part_name, (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        out_parts.append({
            "part": part_name,
            "bbox": [x1, y1, x2, y2],
            "damages": damages,                            # ← แยกเป็นคลัสเตอร์ต่อคลาส (ไม่ merge ทั้งหมดเข้าด้วยกันแบบเดิม)
            "damage_coverage": round(total_cover, 4),      # ← รวมทุกคลาส (union cross-class)
            "damage_coverage_percent": int(round(total_cover * 100.0)),
        })

    # ---------- 7) สร้าง Virtual Parts สำหรับ damage ที่ยังไม่ถูกจับคู่ ----------
    unmatched_damages = []
    for j, damage_box in enumerate(damage_result.boxes):
        if j in matched_damage_indices:
            continue
        damage_cls = int(damage_box.cls.item())
        damage_name = _DAMAGE_NAMES[damage_cls]
        confidence = float(damage_box.conf.item())
        dmask = damage_masks[j]
        x1, y1, x2, y2 = map(int, damage_box.xyxy[0].tolist())
        unmatched_damages.append({
            "id": j,
            "class": damage_name,
            "confidence": confidence,
            "mask": dmask,
            "bbox": [x1, y1, x2, y2]
        })

    # ✅ 7.1) cluster unmatched โดยไม่จำกัดชนิด (รวม damage ทุก class ที่อยู่ใกล้กัน)
    clusters: List[List[Dict[str, Any]]] = []
    visited: set[int] = set()

    def bbox_center(bbox):
        """หาจุดศูนย์กลาง bbox"""
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    for i, d1 in enumerate(unmatched_damages):
        if i in visited:
            continue
        cluster = [d1]
        visited.add(i)

        for j, d2 in enumerate(unmatched_damages):
            if j in visited:
                continue
            iou = mask_iou(d1["mask"], d2["mask"])
            (cx1, cy1), (cx2, cy2) = bbox_center(d1["bbox"]), bbox_center(d2["bbox"])
            dx, dy = abs(cx1 - cx2), abs(cy1 - cy2)

            # ✅ รวมทุกชนิดที่ทับหรือใกล้กันมากพอ
            if (iou >= prox_iou_unmatched) or (dx <= prox_dx_unmatched and dy <= prox_dy_unmatched):
                cluster.append(d2)
                visited.add(j)

        clusters.append(cluster)

    # ✅ 7.2) สร้าง Virtual Part สำหรับแต่ละคลัสเตอร์ที่รวมแล้ว
    for idx, group in enumerate(clusters):
        if not group:
            continue

        xs1, ys1, xs2, ys2 = [], [], [], []
        damage_union_all = np.zeros((H, W), dtype=bool)
        damages_summary: List[Dict[str, Any]] = []

        for g in group:
            x1, y1, x2, y2 = g["bbox"]
            xs1.append(x1); ys1.append(y1)
            xs2.append(x2); ys2.append(y2)
            damage_union_all |= g["mask"]
            damages_summary.append({
                "class": g["class"],
                "confidence": round(g["confidence"], 4),
                "count": 1,
                "mask_iou": None,
                "mask_coverage": 1.0
            })

        bbox_merged = [min(xs1), min(ys1), max(xs2), max(ys2)]
        total_cover = 1.0  # สำหรับ virtual part แสดงเต็มคลัสเตอร์

        if render_overlay and overlay_bgr is not None:
            mask_uint8 = (damage_union_all.astype(np.uint8) * 255)
            colored = np.zeros_like(overlay_bgr, dtype=np.uint8)
            color = np.array([160, 160, 160], dtype=np.uint8)  # สีเทาอ่อน
            for c in range(3):
                colored[:, :, c] = (mask_uint8 * int(color[c])) // 255
            overlay_bgr = cv2.addWeighted(overlay_bgr, 1.0, colored, 0.30, 0.0)
            cv2.rectangle(overlay_bgr, (bbox_merged[0], bbox_merged[1]),
                          (bbox_merged[2], bbox_merged[3]), (160, 160, 160), 2)
            cv2.putText(overlay_bgr, f"ไม่พบชิ้นส่วน-{idx+1}", (bbox_merged[0], max(12, bbox_merged[1]-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 2)

        out_parts.append({
            "part": "",
            "is_virtual": True,
            "bbox": bbox_merged,
            "damages": damages_summary,
            "damage_coverage": round(total_cover, 4),
            "damage_coverage_percent": int(round(total_cover * 100.0)),
        })


    # ---------- 8) รูป overlay (ถ้าขอ) ----------
    out: Dict[str, Any] = {"ok": True, "width": W, "height": H, "parts": out_parts}
    if render_overlay and overlay_bgr is not None:
        ok, buf = cv2.imencode(".jpg", overlay_bgr)
        if ok:
            out["overlay_image_b64"] = base64.b64encode(buf).decode("utf-8")
            out["overlay_mime"] = "image/jpeg"
    return out
