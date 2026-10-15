# CVAT YOLO 1.1 Export -> Merged Class-folder Crop Export
# Local PC 

import os
import json
import zipfile
import shutil
import csv
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image

# ============================================================
# 1) AYARLAR
# ============================================================

ZIP_PATH = r"C:\Users\ahmet\Desktop\yeni_yololar\18kasim_mina3.zip"

OUT_DIR = r"C:\Users\ahmet\Desktop\yeni_yololar\18kasim_mina3"

WORK_DIR = r"C:\Users\ahmet\Desktop\yeni_yololar\temp_yolo_unzipped"

# Kutuların etrafına biraz pay eklemek için
# 0.00 yaparsan tam kutudan crop alır
PADDING_RATIO = 0.03

# Çıktı klasörü varsa silip baştan oluştursun mu?
OVERWRITE_OUT_DIR = True

# Çok küçük crop'ları da kaydetmek için minimum kontrolü sadece geçersiz kutuda yapıyoruz
MIN_VALID_SIZE = 1

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ============================================================
# SADECE BU SINIFLARI KAYDET + BIRLESTIRME HARITASI
# ============================================================
# Sol taraf: obj.names icindeki orijinal CVAT/YOLO sinifi
# Sag taraf: crop'un kaydedilecegi nihai klasor adi
#
# Bu haritada olmayan siniflar otomatik atlanir.
# Ornek: healthy, pes, fruit, flower, stem siniflari kaydedilmez.

MERGE_MAP = {
    # flies front leaf -> flies_leaf
    "flies_up_leaf": "flies_leaf",
    "flies_mid_leaf": "flies_leaf",
    "flies_low_leaf": "flies_leaf",

    # flies back leaf -> flies_back_leaf
    "flies_up_back_leaf": "flies_back_leaf",
    "flies_mid_back_leaf": "flies_back_leaf",
    "flies_low_back_leaf": "flies_back_leaf",

    # trips front leaf -> trips_leaf
    "trips_up_leaf": "trips_leaf",
    "trips_mid_leaf": "trips_leaf",
    "trips_low_leaf": "trips_leaf",

    # trips back leaf -> trips_back_leaf
    "trips_up_back_leaf": "trips_back_leaf",
    "trips_mid_back_leaf": "trips_back_leaf",
    "trips_low_back_leaf": "trips_back_leaf",

    # tuta front leaf -> tuta_leaf
    "tuta_up_leaf": "tuta_leaf",
    "tuta_mid_leaf": "tuta_leaf",
    "tuta_low_leaf": "tuta_leaf",

    # tuta back leaf -> tuta_back_leaf
    "tuta_up_back_leaf": "tuta_back_leaf",
    "tuta_mid_back_leaf": "tuta_back_leaf",
    "tuta_low_back_leaf": "tuta_back_leaf",

    # nutdef front leaf -> nutdef_leaf
    "nutdef_up_leaf": "nutdef_leaf",
    "nutdef_mid_leaf": "nutdef_leaf",
    "nutdef_low_leaf": "nutdef_leaf",

    # nutdef back leaf -> nutdef_back_leaf
    "nutdef_up_back_leaf": "nutdef_back_leaf",
    "nutdef_mid_back_leaf": "nutdef_back_leaf",
    "nutdef_low_back_leaf": "nutdef_back_leaf",
}



# ============================================================
# 2) YARDIMCI FONKSİYONLAR
# ============================================================

def reset_dir(path):
    path = Path(path)
    if path.exists() and OVERWRITE_OUT_DIR:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_name(name):
    bad_chars = '<>:"/\\|?*'
    cleaned = "".join("_" if c in bad_chars else c for c in str(name))
    return cleaned.strip() or "unknown"


def find_file(root, filename):
    root = Path(root)
    matches = list(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"{filename} bulunamadı.")
    return matches[0]


def load_classes(obj_names_path):
    with open(obj_names_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]

    if not classes:
        raise RuntimeError("obj.names boş görünüyor.")

    return classes


def yolo_to_xyxy(xc, yc, bw, bh, img_w, img_h, padding_ratio=0.0):
    """
    YOLO formatı:
    x_center, y_center, width, height normalize değerlerdir.
    Bunları pixel x1, y1, x2, y2 değerlerine çevirir.
    """

    x1 = (xc - bw / 2) * img_w
    y1 = (yc - bh / 2) * img_h
    x2 = (xc + bw / 2) * img_w
    y2 = (yc + bh / 2) * img_h

    box_w = x2 - x1
    box_h = y2 - y1

    x1 -= box_w * padding_ratio
    x2 += box_w * padding_ratio
    y1 -= box_h * padding_ratio
    y2 += box_h * padding_ratio

    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(img_w, int(round(x2)))
    y2 = min(img_h, int(round(y2)))

    return x1, y1, x2, y2


def unique_path(path):
    """
    Aynı isimli dosya varsa üzerine yazmasın diye sonuna _1, _2 ekler.
    """
    path = Path(path)

    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    i = 1
    while True:
        new_path = parent / f"{stem}_{i}{suffix}"
        if not new_path.exists():
            return new_path
        i += 1


def resolve_train_image_path(root, train_line):
    """
    train.txt içinde genelde:
    data/obj_train_data/IMG_x.jpg

    Zip açıldıktan sonra bazen gerçek yol:
    /content/yolo_export_unzipped/obj_train_data/IMG_x.jpg

    olabilir. Bu fonksiyon ikisini de dener.
    """
    root = Path(root)
    train_line = train_line.strip()

    candidates = []

    # Direkt root ile birleştir
    candidates.append(root / train_line)

    # Baştaki data/ kısmını kaldırarak dene
    if train_line.startswith("data/"):
        candidates.append(root / train_line.replace("data/", "", 1))

    # Sadece dosya adına göre obj_train_data içinde dene
    filename = Path(train_line).name
    candidates.extend(list(root.rglob(filename)))

    for c in candidates:
        if c.exists():
            return c

    return None


# ============================================================
# 3) ZIP'İ AÇ
# ============================================================

if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)

os.makedirs(WORK_DIR, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(WORK_DIR)

root = Path(WORK_DIR)

print("Zip açıldı:", root)


# ============================================================
# 4) SINIFLARI OKU
# ============================================================

obj_names_path = find_file(root, "obj.names")
classes = load_classes(obj_names_path)

print("\nobj.names:", obj_names_path)
print("Sınıf sayısı:", len(classes))

print("\nSınıf ID sıralaması:")
for i, cls in enumerate(classes):
    print(f"{i}: {cls}")


# ============================================================
# 5) GÖRSELLERİ BUL
# ============================================================

train_candidates = list(root.rglob("train.txt"))

images = []

if train_candidates:
    train_path = train_candidates[0]
    print("\ntrain.txt bulundu:", train_path)

    with open(train_path, "r", encoding="utf-8") as f:
        train_lines = [line.strip() for line in f.readlines() if line.strip()]

    for line in train_lines:
        img_path = resolve_train_image_path(root, line)
        if img_path is not None and img_path.suffix.lower() in IMAGE_EXTS:
            images.append(img_path)

    print("train.txt üzerinden bulunan görsel sayısı:", len(images))

else:
    print("\ntrain.txt bulunamadı, obj_train_data klasörü aranıyor.")

# train.txt ile image bulunamazsa obj_train_data içinden tara
if not images:
    obj_train_dirs = list(root.rglob("obj_train_data"))

    if obj_train_dirs:
        img_root = obj_train_dirs[0]
    else:
        img_root = root

    images = sorted([
        p for p in img_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ])

    print("Klasör taramasıyla bulunan görsel sayısı:", len(images))

images = sorted(list(set(images)), key=lambda p: str(p).lower())

if not images:
    raise RuntimeError("Hiç görsel bulunamadı.")

print("\nİlk 5 görsel:")
for p in images[:5]:
    print(" -", p)


# ============================================================
# 6) CROP EXPORT
# ============================================================

reset_dir(OUT_DIR)
OUT_DIR = Path(OUT_DIR)

counts = Counter()
skipped = Counter()
summary_rows = []

for img_no, img_path in enumerate(images, start=1):

    # Senin yapında label dosyası aynı klasörde aynı isimli .txt
    label_path = img_path.with_suffix(".txt")

    if not label_path.exists():
        skipped["missing_label_file"] += 1
        continue

    # 0 byte txt = bu resimde annotation yok
    if label_path.stat().st_size == 0:
        skipped["empty_label_file"] += 1
        continue

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        skipped["image_open_error"] += 1
        print("Görsel açılamadı:", img_path, e)
        continue

    img_w, img_h = img.size

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if not lines:
        skipped["empty_label_file"] += 1
        img.close()
        continue

    for obj_idx, line in enumerate(lines, start=1):
        parts = line.split()

        if len(parts) < 5:
            skipped["bad_yolo_line"] += 1
            continue

        try:
            class_id = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
        except Exception:
            skipped["parse_error"] += 1
            continue

        if class_id < 0 or class_id >= len(classes):
            skipped["bad_class_id"] += 1
            continue

        original_label = classes[class_id]

        # Bu sinif secili degilse hic kaydetme
        if original_label not in MERGE_MAP:
            skipped["class_not_selected"] += 1
            continue

        # Secili sinifi nihai/merged sinifa cevir
        label = MERGE_MAP[original_label]

        x1, y1, x2, y2 = yolo_to_xyxy(
            xc, yc, bw, bh,
            img_w, img_h,
            padding_ratio=PADDING_RATIO
        )

        crop_w = x2 - x1
        crop_h = y2 - y1

        if crop_w < MIN_VALID_SIZE or crop_h < MIN_VALID_SIZE:
            skipped["invalid_box"] += 1
            continue

        crop = img.crop((x1, y1, x2, y2))

        label_dir = OUT_DIR / safe_name(label)
        label_dir.mkdir(parents=True, exist_ok=True)

        out_name = (
            f"{safe_name(label)}"
            f"__from_{safe_name(original_label)}"
            f"__{img_path.stem}"
            f"__obj{obj_idx:04d}"
            f"__{crop_w}x{crop_h}.jpg"
        )

        out_path = unique_path(label_dir / out_name)

        crop.save(out_path, quality=95)

        counts[label] += 1

        summary_rows.append({
            "label": label,
            "original_label": original_label,
            "merged_label": label,
            "class_id": class_id,
            "source_image": str(img_path),
            "label_file": str(label_path),
            "output_path": str(out_path),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": crop_w,
            "height": crop_h,
            "yolo_x_center": xc,
            "yolo_y_center": yc,
            "yolo_width": bw,
            "yolo_height": bh,
        })

    img.close()

    if img_no % 50 == 0:
        print(f"{img_no}/{len(images)} görsel işlendi...")


# ============================================================
# 7) ÖZET CSV
# ============================================================

summary_csv = OUT_DIR / "export_summary.csv"

with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "label",
        "original_label",
        "merged_label",
        "class_id",
        "source_image",
        "label_file",
        "output_path",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
        "yolo_x_center",
        "yolo_y_center",
        "yolo_width",
        "yolo_height",
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)


# ============================================================
# 8) SONUÇ
# ============================================================

print("\nExport tamamlandı.")
print("Çıktı klasörü:", OUT_DIR)
print("Özet CSV:", summary_csv)

print("\nToplam kaydedilen crop:", sum(counts.values()))

print("\nSınıf bazlı crop sayıları:")
for label, count in counts.most_common():
    print(f"{label}: {count}")

print("\nAtlananlar:")
for reason, count in skipped.items():
    print(f"{reason}: {count}")

from pathlib import Path

CHECK_DIR = OUT_DIR
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ============================================================
# SADECE BU SINIFLARI KAYDET + BIRLESTIRME HARITASI
# ============================================================
# Sol taraf: obj.names icindeki orijinal CVAT/YOLO sinifi
# Sag taraf: crop'un kaydedilecegi nihai klasor adi
#
# Bu haritada olmayan siniflar otomatik atlanir.
# Ornek: healthy, pes, fruit, flower, stem siniflari kaydedilmez.

MERGE_MAP = {
    # flies front leaf -> flies_leaf
    "flies_up_leaf": "flies_leaf",
    "flies_mid_leaf": "flies_leaf",
    "flies_low_leaf": "flies_leaf",

    # flies back leaf -> flies_back_leaf
    "flies_up_back_leaf": "flies_back_leaf",
    "flies_mid_back_leaf": "flies_back_leaf",
    "flies_low_back_leaf": "flies_back_leaf",

    # trips front leaf -> trips_leaf
    "trips_up_leaf": "trips_leaf",
    "trips_mid_leaf": "trips_leaf",
    "trips_low_leaf": "trips_leaf",

    # trips back leaf -> trips_back_leaf
    "trips_up_back_leaf": "trips_back_leaf",
    "trips_mid_back_leaf": "trips_back_leaf",
    "trips_low_back_leaf": "trips_back_leaf",

    # tuta front leaf -> tuta_leaf
    "tuta_up_leaf": "tuta_leaf",
    "tuta_mid_leaf": "tuta_leaf",
    "tuta_low_leaf": "tuta_leaf",

    # tuta back leaf -> tuta_back_leaf
    "tuta_up_back_leaf": "tuta_back_leaf",
    "tuta_mid_back_leaf": "tuta_back_leaf",
    "tuta_low_back_leaf": "tuta_back_leaf",

    # nutdef front leaf -> nutdef_leaf
    "nutdef_up_leaf": "nutdef_leaf",
    "nutdef_mid_leaf": "nutdef_leaf",
    "nutdef_low_leaf": "nutdef_leaf",

    # nutdef back leaf -> nutdef_back_leaf
    "nutdef_up_back_leaf": "nutdef_back_leaf",
    "nutdef_mid_back_leaf": "nutdef_back_leaf",
    "nutdef_low_back_leaf": "nutdef_back_leaf",
}


root = Path(CHECK_DIR)

class_dirs = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")]
total_images = sum(
    1
    for d in class_dirs
    for p in d.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS
)

print("Klasör sayısı:", len(class_dirs))
print("Toplam resim:", total_images)

