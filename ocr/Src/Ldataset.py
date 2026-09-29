import pathlib
import tensorflow as tf

# ==========================================
# ۱. تنظیم کاراکترها و دیکشنری (Vocabulary)
# ==========================================
# لیست کامل کاراکترهای ممکن در پلاک‌ها (اعداد فارسی + حروف)
DIGITS = ["۰", "۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹"]
LETTERS = [
    "الف", "ب", "پ", "ت", "ث", "ج", "چ", "ح", "خ", "د", "ذ", "ر",
    "ز", "ژ", "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ", "ف", "ق",
    "ک", "گ", "ل", "م", "ن", "و", "ه", "ی"
]

VOCABULARY = DIGITS + LETTERS
MAX_LABEL_LEN = 10  # طول دنباله (مثلاً ۲ رقم + ۱ حرف + ۳ رقم + ۲ رقم + فاصله)

# ساخت لایه StringLookup برای نگاشت کاراکترها به عدد
char_to_num = tf.keras.layers.StringLookup(
    vocabulary=VOCABULARY, mask_token=None, num_oov_indices=1  # index 0 reserved/OOV
)

# ==========================================
# ۲. تابع خواندن labels.txt
# ==========================================
def load_labels_file(data_dir):
    data_dir = pathlib.Path(data_dir)
    labels_path = data_dir / "labels.txt"
    images_dir = data_dir / "images"

    img_paths = []
    labels = []

    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            img_name = parts[0]
            # حذف فاصله‌های اضافی و یکدست‌سازی لیبل
            label_text = parts[1].replace(" ", "")

            img_paths.append(str(images_dir / img_name))
            labels.append(label_text)

    return img_paths, labels

# ==========================================
# ۳. تابع پیش‌پردازش هر نمونه (Image & Label)
# ==========================================
def process_sample(img_path, label_text):
    # الف) پردازش تصویر
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)  # نرمال‌سازی بین 0.0 تا 1.0
    img = tf.image.resize(img, [32, 160])  # (Height=32, Width=160)

    # ب) پردازش متن و تبدیل به اعداد (Tensor)
    chars = tf.strings.unicode_split(label_text, input_encoding="UTF-8")
    label_encoded = char_to_num(chars)

    # ج) پد کردن (Padding) تا طول ثابت MAX_LABEL_LEN
    pad_len = MAX_LABEL_LEN - tf.shape(label_encoded)[0]
    label_encoded = tf.pad(label_encoded, [[0, pad_len]], constant_values=0)

    return {"image_input": img}, label_encoded

# ==========================================
# ۴. تابع اصلی ساخت tf.data.Dataset
# ==========================================
def get_dataset(data_dir="../data/synthetic_plates_crnn", batch_size=32, validation_split=0.2):
    # ۱. دریافت تمام مسیرها و لیبل‌ها
    img_paths, labels = load_labels_file(data_dir)

    num_samples = len(img_paths)
    val_size = int(num_samples * validation_split)

    # آرایش تصادفی داده‌ها
    indices = tf.range(num_samples)
    indices = tf.random.shuffle(indices, seed=42)

    img_paths = tf.gather(img_paths, indices)
    labels = tf.gather(labels, indices)

    # تقسیم به Train و Validation
    train_paths, val_paths = img_paths[val_size:], img_paths[:val_size]
    train_labels, val_labels = labels[val_size:], labels[:val_size]

    # ساخت دیتاست
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        train_ds.map(process_sample, num_parallel_calls=AUTOTUNE)
        .shuffle(buffer_size=5000)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        val_ds.map(process_sample, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds