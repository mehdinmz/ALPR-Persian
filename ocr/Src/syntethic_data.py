import io
import json
import random
from pathlib import Path

import arabic_reshaper
import numpy as np
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("../data/synthetic_plates_crnn")
FONTS_DIR = Path("../Fonts")

IMAGE_SIZE = (160, 32)
TOTAL_SAMPLES = 50000

FONT_SIZE_RANGE = (18, 22)

ROTATION_RANGE = (-3, 3)
SHIFT_RANGE = (-1, 1)

NOISE_STD_RANGE = (2, 8)

BLUR_PROBABILITY = 0.15
BLUR_RADIUS_RANGE = (0.2, 0.5)

PERSPECTIVE_PROBABILITY = 0.10

BRIGHTNESS_RANGE = (0.85, 1.15)
CONTRAST_RANGE = (0.85, 1.15)

JPEG_ARTIFACT_PROBABILITY = 0.15

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# CHARACTERS
# ============================================================

DIGITS = ("۰", "۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹")

LETTERS = (
    "الف",
    "ب",
    "پ",
    "ت",
    "ث",
    "ج",
    "چ",
    "ح",
    "خ",
    "د",
    "ذ",
    "ر",
    "ز",
    "ژ",
    "س",
    "ش",
    "ص",
    "ض",
    "ط",
    "ظ",
    "ع",
    "غ",
    "ف",
    "ق",
    "ک",
    "گ",
    "ل",
    "م",
    "ن",
    "و",
    "ه",
    "ی",
)

# ============================================================
# FIND & VALIDATE FONTS (فیلتر کردن فونت‌های خراب)
# ============================================================

all_font_paths = [
    p for p in FONTS_DIR.rglob("*") if p.suffix.lower() in {".ttf", ".otf"}
]

if not all_font_paths:
    raise FileNotFoundError(
        f"No .ttf or .otf fonts found in:\n{FONTS_DIR.resolve()}"
    )


def validate_fonts(font_list):
    """حذف فونت‌هایی که قابلیت رندر حروف فارسی یا اعداد را ندارند"""
    valid_fonts = []
    test_canvas = Image.new("L", (100, 30), 255)
    test_draw = ImageDraw.Draw(test_canvas)

    for p in font_list:
        try:
            f = ImageFont.truetype(str(p), 20)
            # رندر تست
            bbox = test_draw.textbbox((0, 0), "۱۲ ب ۳۴۵", font=f)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            # اگر فونت ابعاد خروجی معتبری داد، آن را می‌پذیریم
            if w > 10 and h > 5:
                valid_fonts.append(p)
        except Exception:
            continue
    return valid_fonts


valid_font_paths = validate_fonts(all_font_paths)

if not valid_font_paths:
    raise RuntimeError(
        "هیچ فونت معتبری برای رندر حروف و اعداد فارسی یافت نشد!"
    )

print(
    f"Found {len(all_font_paths)} total fonts -> {len(valid_font_paths)} valid fonts loaded."
)


def random_font():
    font_path = random.choice(valid_font_paths)
    font_size = random.randint(FONT_SIZE_RANGE[0], FONT_SIZE_RANGE[1])
    return ImageFont.truetype(str(font_path), font_size)


def shape_persian_word(word):
    reshaped = arabic_reshaper.reshape(word)
    return get_display(reshaped)


# ============================================================
# RENDER SEQUENCE
# ============================================================


def render_text_sequence(d1, letter, d2, city_code):
    font = random_font()
    canvas = Image.new("L", IMAGE_SIZE, 255)
    draw = ImageDraw.Draw(canvas)

    part1 = d1
    part2 = shape_persian_word(letter)
    part3 = d2
    part4 = city_code

    bbox1 = draw.textbbox((0, 0), part1, font=font)
    w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]

    bbox2 = draw.textbbox((0, 0), part2, font=font)
    w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]

    bbox3 = draw.textbbox((0, 0), part3, font=font)
    w3, h3 = bbox3[2] - bbox3[0], bbox3[3] - bbox3[1]

    bbox4 = draw.textbbox((0, 0), part4, font=font)
    w4, h4 = bbox4[2] - bbox4[0], bbox4[3] - bbox4[1]

    space_w = 5
    sep_space = 7

    total_width = w1 + space_w + w2 + space_w + w3 + (sep_space * 2) + w4
    start_x = (IMAGE_SIZE[0] - total_width) // 2

    # ۱. رسم ۲ رقم اول
    y1 = (IMAGE_SIZE[1] - h1) // 2 - bbox1[1]
    draw.text((start_x - bbox1[0], y1), part1, font=font, fill=0)

    # ۲. رسم حرف
    x_letter = start_x + w1 + space_w
    y2 = (IMAGE_SIZE[1] - h2) // 2 - bbox2[1]
    draw.text((x_letter - bbox2[0], y2), part2, font=font, fill=0)

    # ۳. رسم ۳ رقم
    x_d2 = x_letter + w2 + space_w
    y3 = (IMAGE_SIZE[1] - h3) // 2 - bbox3[1]
    draw.text((x_d2 - bbox3[0], y3), part3, font=font, fill=0)

    # ۴. رسم خط جداکننده کادر شهر
    x_line = x_d2 + w3 + sep_space
    draw.line([(x_line, 4), (x_line, IMAGE_SIZE[1] - 4)], fill=0, width=2)

    # ۵. رسم کد شهر
    x_city = x_line + sep_space
    y4 = (IMAGE_SIZE[1] - h4) // 2 - bbox4[1]
    draw.text((x_city - bbox4[0], y4), part4, font=font, fill=0)

    return canvas


# ============================================================
# SAFE AUGMENTATIONS
# ============================================================


def add_rotation_and_shift(image):
    angle = random.uniform(ROTATION_RANGE[0], ROTATION_RANGE[1])
    image = image.rotate(
        angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=255
    )

    shift_x = random.randint(SHIFT_RANGE[0], SHIFT_RANGE[1])
    shift_y = random.randint(SHIFT_RANGE[0], SHIFT_RANGE[1])

    shifted = Image.new("L", image.size, 255)
    shifted.paste(image, (shift_x, shift_y))
    return shifted


def add_perspective(image):
    if random.random() > PERSPECTIVE_PROBABILITY:
        return image
    width, height = image.size
    max_x = int(width * 0.015)
    max_y = int(height * 0.015)

    coefficients = (
        1,
        random.uniform(-0.001, 0.001),
        random.uniform(-max_x, max_x),
        random.uniform(-0.001, 0.001),
        1,
        random.uniform(-max_y, max_y),
        random.uniform(-0.00001, 0.00001),
        random.uniform(-0.00001, 0.00001),
    )
    try:
        image = image.transform(
            image.size,
            Image.Transform.PERSPECTIVE,
            coefficients,
            resample=Image.Resampling.BICUBIC,
            fillcolor=255,
        )
    except Exception:
        pass
    return image


def add_blur(image):
    if random.random() > BLUR_PROBABILITY:
        return image
    radius = random.uniform(BLUR_RADIUS_RANGE[0], BLUR_RADIUS_RANGE[1])
    return image.filter(ImageFilter.GaussianBlur(radius))


def add_brightness_contrast(image):
    brightness = random.uniform(BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1])
    contrast = random.uniform(CONTRAST_RANGE[0], CONTRAST_RANGE[1])

    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    return image


def add_noise(image):
    image_array = np.asarray(image, dtype=np.float32)
    noise_std = random.uniform(NOISE_STD_RANGE[0], NOISE_STD_RANGE[1])
    noise = np.random.normal(0, noise_std, image_array.shape)

    image_array += noise
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    return Image.fromarray(image_array, mode="L")


def add_jpeg_artifacts(image):
    if random.random() > JPEG_ARTIFACT_PROBABILITY:
        return image
    buffer = io.BytesIO()
    quality = random.randint(50, 80)
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("L")


def generate_sample(d1, letter, d2, city_code):
    image = render_text_sequence(d1, letter, d2, city_code)
    image = add_rotation_and_shift(image)
    image = add_perspective(image)
    image = add_blur(image)
    image = add_brightness_contrast(image)
    image = add_noise(image)
    image = add_jpeg_artifacts(image)
    return image


# ============================================================
# DATASET GENERATION
# ============================================================


def generate_dataset():
    images_dir = OUTPUT_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    labels_path = OUTPUT_DIR / "labels.txt"

    print("=" * 60)
    print("GENERATING SYNTHETIC PLATE DATASET FOR CRNN")
    print("=" * 60)
    print(f"Total samples : {TOTAL_SAMPLES:,}")
    print(f"Image size    : {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
    print("=" * 60)

    with open(labels_path, "w", encoding="utf-8") as f_labels:
        for i in range(TOTAL_SAMPLES):
            d1 = "".join(random.choices(DIGITS, k=2))
            letter = random.choice(LETTERS)
            d2 = "".join(random.choices(DIGITS, k=3))
            city_code = "".join(random.choices(DIGITS, k=2))

            # بررسی و اطمینان سخت‌گیرانه از عدم وجود تصویر سفید
            attempts = 0
            while True:
                image = generate_sample(d1, letter, d2, city_code)
                img_array = np.array(image)
                # پیکسل‌های تیره باید حتماً وجود داشته باشند (متن رسم شده باشد)
                if np.min(img_array) < 180 or attempts > 10:
                    break
                attempts += 1

            img_name = f"plate_{i:06d}.png"
            img_path = images_dir / img_name
            image.save(img_path, format="PNG")

            label_text = f"{d1} {letter} {d2} {city_code}"
            f_labels.write(f"{img_name},{label_text}\n")

            if (i + 1) % 5000 == 0 or (i + 1) == TOTAL_SAMPLES:
                print(f"Generated [{i + 1:,}/{TOTAL_SAMPLES:,}] samples...")

    print("=" * 60)
    print("COMPLETE!")
    print(f"Images saved in : {images_dir}")
    print(f"Labels saved in : {labels_path}")
    print("=" * 60)


if __name__ == "__main__":
    generate_dataset()