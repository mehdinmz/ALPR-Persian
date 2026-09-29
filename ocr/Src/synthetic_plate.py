import io
import random
from pathlib import Path

import arabic_reshaper
import numpy as np
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


# ============================================================
# CONFIGURATION
# ============================================================

# مسیرها بر اساس خود فایل محاسبه می‌شوند، نه محل اجرای terminal
OCR_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = OCR_ROOT / "data" / "synthetic_plates_crnn"
FONTS_DIR = OCR_ROOT / "Fonts"

# اندازه نهایی مورد استفاده CRNN
IMAGE_SIZE = (160, 32)

# برای اینکه رندر حروف کیفیت بهتری داشته باشد،
# ابتدا با رزولوشن بالاتر ساخته و سپس resize می‌کنیم.
RENDER_SCALE = 4

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

DIGITS = (
    "۰",
    "۱",
    "۲",
    "۳",
    "۴",
    "۵",
    "۶",
    "۷",
    "۸",
    "۹",
)

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
# REGIONAL / CITY CODES
# ============================================================

# پروژه فعلی mapping کدهای مناطق ایران را ندارد.
# بنابراین فعلاً کدهای 48 و 58 برای بوشهر استفاده شده‌اند.
#
# اگر بعداً دیتاست کل ایران ساختی، این tuple را با تمام
# کدهای معتبر منطقه‌ای موردنظرت جایگزین کن.
#
# نکته مهم:
# تولید تصادفی 00 تا 99 برای دیتاستی که قرار است شبیه
# پلاک واقعی باشد کار درستی نیست، چون بسیاری از این ترکیب‌ها
# در دنیای واقعی کد منطقه‌ای معتبر نیستند.

CITY_CODES = (
    "۴۸",
    "۵۸",
)


# ============================================================
# FIND & VALIDATE FONTS
# ============================================================

all_font_paths = [
    p
    for p in FONTS_DIR.rglob("*")
    if p.suffix.lower() in {".ttf", ".otf"}
]

if not all_font_paths:
    raise FileNotFoundError(
        f"No .ttf or .otf fonts found in:\n"
        f"{FONTS_DIR.resolve()}"
    )


def validate_fonts(font_list):
    """
    فونت‌هایی که قابلیت رندر فارسی و اعداد را ندارند حذف می‌شوند.
    """

    valid_fonts = []

    test_canvas = Image.new(
        "L",
        (300, 80),
        255
    )

    test_draw = ImageDraw.Draw(
        test_canvas
    )

    for font_path in font_list:

        try:
            font = ImageFont.truetype(
                str(font_path),
                20
            )

            test_text = "۱۲ الف ۳۴۵ ۴۸"

            bbox = test_draw.textbbox(
                (0, 0),
                test_text,
                font=font
            )

            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            if width > 20 and height > 5:
                valid_fonts.append(
                    font_path
                )

        except Exception:
            continue

    return valid_fonts


valid_font_paths = validate_fonts(
    all_font_paths
)

if not valid_font_paths:
    raise RuntimeError(
        "هیچ فونت معتبری برای رندر حروف و اعداد فارسی پیدا نشد!"
    )

print(
    f"Found {len(all_font_paths)} total fonts -> "
    f"{len(valid_font_paths)} valid fonts loaded."
)


def random_font():
    """
    انتخاب تصادفی یک فونت معتبر با سایز تصادفی.
    """

    font_path = random.choice(
        valid_font_paths
    )

    font_size = random.randint(
        FONT_SIZE_RANGE[0],
        FONT_SIZE_RANGE[1]
    )

    return ImageFont.truetype(
        str(font_path),
        font_size * RENDER_SCALE
    )


def shape_persian_word(word):
    """
    آماده‌سازی حروف فارسی برای رندر صحیح در PIL.

    مهم:
    این فقط rendering را تغییر می‌دهد.
    label اصلی همچنان همان "الف" است.
    """

    reshaped = arabic_reshaper.reshape(
        word
    )

    return get_display(
        reshaped
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def text_bbox(draw, text, font):
    bbox = draw.textbbox(
        (0, 0),
        text,
        font=font
    )

    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    return bbox, width, height


def fit_font(
    draw,
    text,
    max_width,
    base_font_size
):
    """
    بزرگ‌ترین فونتی را پیدا می‌کند که داخل slot قرار بگیرد.
    """

    font_path = random.choice(
        valid_font_paths
    )

    for size in range(
        int(base_font_size),
        7,
        -1
    ):

        font = ImageFont.truetype(
            str(font_path),
            size * RENDER_SCALE
        )

        _, width, _ = text_bbox(
            draw,
            text,
            font
        )

        if width <= max_width:
            return font

    return ImageFont.truetype(
        str(font_path),
        8 * RENDER_SCALE
    )


# ============================================================
# PLATE DATA GENERATION
# ============================================================

def generate_plate_fields():
    """
    ساخت یک پلاک با ساختار:

        ۱۲ الف ۳۴۵ ۴۸

    اجزا:

        ۲ رقم اول
        ۱ حرف
        ۳ رقم وسط
        ۲ رقم منطقه‌ای سمت راست
    """

    first_two_digits = "".join(
        random.choices(
            DIGITS,
            k=2
        )
    )

    letter = random.choice(
        LETTERS
    )

    middle_three_digits = "".join(
        random.choices(
            DIGITS,
            k=3
        )
    )

    city_code = random.choice(
        CITY_CODES
    )

    return (
        first_two_digits,
        letter,
        middle_three_digits,
        city_code
    )


# ============================================================
# RENDER TEXT SEQUENCE
# ============================================================

def render_text_sequence(
    d1,
    letter,
    d2,
    city_code
):
    """
    رندر پلاک.

    layout نهایی:

        | BLUE | ۱۲ | الف | ۳۴۵ | ۴۸ |

    بخش سمت راست کاملاً مستقل است.

    این برخلاف نسخه قبلی است که کل متن را یک‌جا
    وسط canvas حساب می‌کرد و ممکن بود city_code
    به لبه بچسبد یا از کادر خارج شود.
    """

    width, height = IMAGE_SIZE

    render_width = (
        width * RENDER_SCALE
    )

    render_height = (
        height * RENDER_SCALE
    )

    # ========================================================
    # Canvas
    # ========================================================

    canvas = Image.new(
        "L",
        (
            render_width,
            render_height
        ),
        245
    )

    draw = ImageDraw.Draw(
        canvas
    )

    def S(value):
        return int(
            round(
                value * RENDER_SCALE
            )
        )

    # ========================================================
    # Plate border
    # ========================================================

    draw.rounded_rectangle(
        [
            S(0.8),
            S(0.8),
            S(width - 0.8),
            S(height - 0.8),
        ],
        radius=S(2),
        outline=25,
        width=S(1)
    )

    # ========================================================
    # Left blue area
    # ========================================================

    blue_left = 1
    blue_right = 16

    draw.rectangle(
        [
            S(blue_left),
            S(1),
            S(blue_right),
            S(height - 1),
        ],
        fill=225
    )

    # مرز سمت راست ناحیه آبی
    draw.line(
        [
            (S(blue_right), S(2)),
            (
                S(blue_right),
                S(height - 2)
            ),
        ],
        fill=100,
        width=S(1)
    )

    # ========================================================
    # FIXED SLOTS
    # ========================================================
    #
    # مختصات بر اساس تصویر 160x32 هستند.
    #
    # main plate:
    #
    #  21----48   52------82   86------123
    #   ۱۲          الف            ۳۴۵
    #
    #                           128
    #                            │
    #                           132---158
    #                              ۴۸
    #
    # ========================================================

    slot_first = (
        21,
        48
    )

    slot_letter = (
        52,
        82
    )

    slot_middle = (
        86,
        123
    )

    separator_x = 128

    slot_city = (
        132,
        158
    )

    # ========================================================
    # Texts
    # ========================================================

    part1 = d1

    part2 = shape_persian_word(
        letter
    )

    part3 = d2

    part4 = city_code

    # ========================================================
    # Fonts
    # ========================================================

    base_font_size = random.randint(
        FONT_SIZE_RANGE[0],
        FONT_SIZE_RANGE[1]
    )

    font1 = fit_font(
        draw,
        part1,
        S(
            slot_first[1]
            - slot_first[0]
        ),
        base_font_size
    )

    font2 = fit_font(
        draw,
        part2,
        S(
            slot_letter[1]
            - slot_letter[0]
        ),
        base_font_size
    )

    font3 = fit_font(
        draw,
        part3,
        S(
            slot_middle[1]
            - slot_middle[0]
        ),
        base_font_size
    )

    font4 = fit_font(
        draw,
        part4,
        S(
            slot_city[1]
            - slot_city[0]
        ),
        base_font_size - 1
    )

    # ========================================================
    # Center text inside slot
    # ========================================================

    def draw_centered_in_slot(
        text,
        font,
        x1,
        x2
    ):
        bbox = draw.textbbox(
            (0, 0),
            text,
            font=font
        )

        text_width = (
            bbox[2] - bbox[0]
        )

        text_height = (
            bbox[3] - bbox[1]
        )

        slot_width = S(
            x2 - x1
        )

        # center horizontal
        x = (
            S(x1)
            + (
                slot_width
                - text_width
            ) // 2
        )

        # center vertical
        y = (
            S(height) // 2
            - (
                bbox[1]
                + text_height // 2
            )
        )

        draw.text(
            (
                x - bbox[0],
                y
            ),
            text,
            font=font,
            fill=10
        )

    # ========================================================
    # Draw 2 first digits
    # ========================================================

    draw_centered_in_slot(
        part1,
        font1,
        slot_first[0],
        slot_first[1]
    )

    # ========================================================
    # Draw Persian letter
    # ========================================================

    draw_centered_in_slot(
        part2,
        font2,
        slot_letter[0],
        slot_letter[1]
    )

    # ========================================================
    # Draw 3 middle digits
    # ========================================================

    draw_centered_in_slot(
        part3,
        font3,
        slot_middle[0],
        slot_middle[1]
    )

    # ========================================================
    # Separator
    # ========================================================

    draw.line(
        [
            (
                S(separator_x),
                S(4)
            ),
            (
                S(separator_x),
                S(height - 4)
            )
        ],
        fill=20,
        width=S(1)
    )

    # ========================================================
    # Draw 2 regional digits on RIGHT
    # ========================================================

    draw_centered_in_slot(
        part4,
        font4,
        slot_city[0],
        slot_city[1]
    )

    # ========================================================
    # Downsample
    # ========================================================

    canvas = canvas.resize(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    return canvas


# ============================================================
# SAFE AUGMENTATIONS
# ============================================================

def add_rotation_and_shift(image):
    angle = random.uniform(
        ROTATION_RANGE[0],
        ROTATION_RANGE[1]
    )

    image = image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=245
    )

    shift_x = random.randint(
        SHIFT_RANGE[0],
        SHIFT_RANGE[1]
    )

    shift_y = random.randint(
        SHIFT_RANGE[0],
        SHIFT_RANGE[1]
    )

    shifted = Image.new(
        "L",
        image.size,
        245
    )

    shifted.paste(
        image,
        (shift_x, shift_y)
    )

    return shifted


def add_perspective(image):
    if random.random() > PERSPECTIVE_PROBABILITY:
        return image

    width, height = image.size

    max_x = int(
        width * 0.015
    )

    max_y = int(
        height * 0.015
    )

    coefficients = (
        1,
        random.uniform(
            -0.001,
            0.001
        ),
        random.uniform(
            -max_x,
            max_x
        ),

        random.uniform(
            -0.001,
            0.001
        ),
        1,
        random.uniform(
            -max_y,
            max_y
        ),

        random.uniform(
            -0.00001,
            0.00001
        ),

        random.uniform(
            -0.00001,
            0.00001
        ),
    )

    try:
        image = image.transform(
            image.size,
            Image.Transform.PERSPECTIVE,
            coefficients,
            resample=Image.Resampling.BICUBIC,
            fillcolor=245
        )

    except Exception:
        pass

    return image


def add_blur(image):
    if random.random() > BLUR_PROBABILITY:
        return image

    radius = random.uniform(
        BLUR_RADIUS_RANGE[0],
        BLUR_RADIUS_RANGE[1]
    )

    return image.filter(
        ImageFilter.GaussianBlur(
            radius
        )
    )


def add_brightness_contrast(image):
    brightness = random.uniform(
        BRIGHTNESS_RANGE[0],
        BRIGHTNESS_RANGE[1]
    )

    contrast = random.uniform(
        CONTRAST_RANGE[0],
        CONTRAST_RANGE[1]
    )

    image = ImageEnhance.Brightness(
        image
    ).enhance(
        brightness
    )

    image = ImageEnhance.Contrast(
        image
    ).enhance(
        contrast
    )

    return image


def add_noise(image):
    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    noise_std = random.uniform(
        NOISE_STD_RANGE[0],
        NOISE_STD_RANGE[1]
    )

    noise = np.random.normal(
        0,
        noise_std,
        image_array.shape
    )

    image_array += noise

    image_array = np.clip(
        image_array,
        0,
        255
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        image_array,
        mode="L"
    )


def add_jpeg_artifacts(image):
    if (
        random.random()
        > JPEG_ARTIFACT_PROBABILITY
    ):
        return image

    buffer = io.BytesIO()

    quality = random.randint(
        50,
        80
    )

    image.save(
        buffer,
        format="JPEG",
        quality=quality
    )

    buffer.seek(0)

    return Image.open(
        buffer
    ).convert("L")


# ============================================================
# COMPLETE SAMPLE PIPELINE
# ============================================================

def generate_sample(
    d1,
    letter,
    d2,
    city_code
):
    """
    ترتیب دقیق augmentationها همان نسخه اصلی است.
    """

    image = render_text_sequence(
        d1,
        letter,
        d2,
        city_code
    )

    image = add_rotation_and_shift(
        image
    )

    image = add_perspective(
        image
    )

    image = add_blur(
        image
    )

    image = add_brightness_contrast(
        image
    )

    image = add_noise(
        image
    )

    image = add_jpeg_artifacts(
        image
    )

    return image


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset():

    images_dir = (
        OUTPUT_DIR
        / "images"
    )

    images_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    labels_path = (
        OUTPUT_DIR
        / "labels.txt"
    )

    print("=" * 60)
    print(
        "GENERATING SYNTHETIC PLATE DATASET FOR CRNN"
    )
    print("=" * 60)

    print(
        f"OCR root      : "
        f"{OCR_ROOT.resolve()}"
    )

    print(
        f"Fonts dir     : "
        f"{FONTS_DIR.resolve()}"
    )

    print(
        f"Output dir    : "
        f"{OUTPUT_DIR.resolve()}"
    )

    print(
        f"Total samples : "
        f"{TOTAL_SAMPLES:,}"
    )

    print(
        f"Image size    : "
        f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"
    )

    print(
        f"City codes    : "
        f"{', '.join(CITY_CODES)}"
    )

    print("=" * 60)

    with open(
        labels_path,
        "w",
        encoding="utf-8"
    ) as f_labels:

        for i in range(
            TOTAL_SAMPLES
        ):

            # =================================================
            # Generate plate text
            # =================================================

            (
                d1,
                letter,
                d2,
                city_code
            ) = generate_plate_fields()

            # =================================================
            # Generate image
            # =================================================

            attempts = 0

            while True:

                image = generate_sample(
                    d1,
                    letter,
                    d2,
                    city_code
                )

                image_array = np.asarray(
                    image
                )

                # جلوگیری از نمونه کاملاً سفید
                if (
                    np.min(
                        image_array
                    ) < 180
                    or attempts >= 10
                ):
                    break

                attempts += 1

            # =================================================
            # Save image
            # =================================================

            image_name = (
                f"plate_{i:06d}.png"
            )

            image_path = (
                images_dir
                / image_name
            )

            image.save(
                image_path,
                format="PNG"
            )

            # =================================================
            # Save label
            # =================================================
            #
            # format:
            #
            # ۱۲ الف ۳۴۵ ۴۸
            #
            # Ldataset.py فعلی فاصله‌ها را حذف می‌کند.
            #

            label_text = (
                f"{d1} "
                f"{letter} "
                f"{d2} "
                f"{city_code}"
            )

            f_labels.write(
                f"{image_name},"
                f"{label_text}\n"
            )

            # =================================================
            # Progress
            # =================================================

            if (
                (i + 1) % 5000 == 0
                or (i + 1) == TOTAL_SAMPLES
            ):

                print(
                    f"Generated "
                    f"[{i + 1:,}/"
                    f"{TOTAL_SAMPLES:,}] "
                    f"samples..."
                )

    print("=" * 60)
    print("COMPLETE!")

    print(
        f"Images saved in : "
        f"{images_dir.resolve()}"
    )

    print(
        f"Labels saved in : "
        f"{labels_path.resolve()}"
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    generate_dataset()