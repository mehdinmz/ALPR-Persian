from pathlib import Path

import numpy as np
import tensorflow as tf


# ============================================================
# VOCABULARY
# ============================================================

DIGITS = [
    "۰", "۱", "۲", "۳", "۴",
    "۵", "۶", "۷", "۸", "۹"
]

LETTERS = [
    "الف", "ب", "پ", "ت", "ث", "ج", "چ", "ح", "خ", "د",
    "ذ", "ر", "ز", "ژ", "س", "ش", "ص", "ض", "ط", "ظ",
    "ع", "غ", "ف", "ق", "ک", "گ", "ل", "م", "ن", "و",
    "ه", "ی"
]

VOCABULARY = DIGITS + LETTERS

# ID = 1 ... N
# 0 = padding
TOKEN_TO_ID = {
    token: i + 1
    for i, token in enumerate(VOCABULARY)
}

ID_TO_TOKEN = {
    i + 1: token
    for i, token in enumerate(VOCABULARY)
}

MAX_LABEL_LEN = 8


# ============================================================
# LABEL ENCODING
# ============================================================

def encode_label(label_text: str):
    """
    Input example:

        ۱۲ الف ۳۴۵ ۴۸

    تبدیل می‌شود به:

        ['۱', '۲', 'الف', '۳', '۴', '۵', '۴', '۸']

    'الف' یک token است، نه سه character.
    """

    parts = label_text.strip().split()

    if len(parts) != 4:
        raise ValueError(
            f"Invalid label format: {label_text!r}"
        )

    first_two = list(parts[0])
    letter = parts[1]
    middle_three = list(parts[2])
    regional_code = list(parts[3])

    tokens = (
        first_two
        + [letter]
        + middle_three
        + regional_code
    )

    if len(tokens) != MAX_LABEL_LEN:
        raise ValueError(
            f"Expected {MAX_LABEL_LEN} tokens, "
            f"got {len(tokens)} for label {label_text!r}"
        )

    encoded = []

    for token in tokens:
        if token not in TOKEN_TO_ID:
            raise ValueError(
                f"Unknown token {token!r} "
                f"in label {label_text!r}"
            )

        encoded.append(TOKEN_TO_ID[token])

    return np.asarray(
        encoded,
        dtype=np.int32
    )


# ============================================================
# LOAD LABELS
# ============================================================

def load_labels_file(data_dir):
    data_dir = Path(data_dir)

    labels_path = data_dir / "labels.txt"
    images_dir = data_dir / "images"

    if not labels_path.exists():
        raise FileNotFoundError(
            f"labels.txt not found:\n{labels_path.resolve()}"
        )

    img_paths = []
    labels = []

    with open(
        labels_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split(",", 1)

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid labels.txt line "
                    f"{line_number}: {line!r}"
                )

            img_name, label_text = parts

            img_path = images_dir / img_name

            if not img_path.exists():
                raise FileNotFoundError(
                    f"Image referenced in labels.txt "
                    f"does not exist:\n{img_path}"
                )

            encoded_label = encode_label(label_text)

            img_paths.append(str(img_path))
            labels.append(encoded_label)

    return (
        np.asarray(img_paths),
        np.asarray(labels, dtype=np.int32)
    )


# ============================================================
# IMAGE PROCESSING
# ============================================================

def process_sample(img_path, label):

    image = tf.io.read_file(img_path)

    image = tf.io.decode_png(
        image,
        channels=1
    )

    image = tf.image.convert_image_dtype(
        image,
        tf.float32
    )

    image = tf.image.resize(
        image,
        [32, 160]
    )

    return {
        "image_input": image
    }, label


# ============================================================
# DATASET
# ============================================================

def get_dataset(
    data_dir=None,
    batch_size=64,
    validation_split=0.2,
    seed=42
):

    if data_dir is None:
        data_dir = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "synthetic_plates_crnn"
        )

    img_paths, labels = load_labels_file(data_dir)

    num_samples = len(img_paths)

    if num_samples == 0:
        raise RuntimeError(
            "Dataset is empty."
        )

    rng = np.random.default_rng(seed)

    indices = np.arange(num_samples)

    rng.shuffle(indices)

    val_size = int(
        num_samples * validation_split
    )

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_paths = img_paths[train_indices]
    train_labels = labels[train_indices]

    val_paths = img_paths[val_indices]
    val_labels = labels[val_indices]

    train_ds = tf.data.Dataset.from_tensor_slices(
        (
            train_paths,
            train_labels
        )
    )

    val_ds = tf.data.Dataset.from_tensor_slices(
        (
            val_paths,
            val_labels
        )
    )

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        train_ds
        .shuffle(
            min(len(train_paths), 5000),
            seed=seed
        )
        .map(
            process_sample,
            num_parallel_calls=AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        val_ds
        .map(
            process_sample,
            num_parallel_calls=AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds