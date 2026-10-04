from pathlib import Path

import tensorflow as tf

import re
from collections import Counter

from model import build_crnn_model
from Ldataset import (
    get_dataset,
    VOCABULARY,
    MAX_LABEL_LEN,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Dataset" / "generated_plates" / "roya_bold" 

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = MODEL_DIR / "best_crnn_model.keras"


# ============================================================
# DATASET
# ============================================================

train_ds, val_ds = get_dataset(
    data_dir=DATA_DIR,
    batch_size=64,
    validation_split=0.2,
    seed=42,
)

pattern = re.compile(r"^(\d{2})(.+?)(\d{3})_(\d{2})$")

letter_tokens = []

for plate_dir in DATA_DIR.iterdir():
    if not plate_dir.is_dir():
        continue

    match = pattern.match(plate_dir.name)

    if not match:
        print("Invalid:", plate_dir.name)
        continue

    first_two, letter, middle_three, city_code = match.groups()
    letter_tokens.append(letter)

counter = Counter(letter_tokens)

print("Unique letter tokens:", len(counter))
print(sorted(counter.keys()))

# ============================================================
# MODEL
# ============================================================

# 43 توکن:
# 10 digits + 33 letters
#
# + 1 CTC blank
#
# => 44 classes

NUM_CLASSES = len(VOCABULARY) + 2

model = build_crnn_model(
    input_shape=(32, 160, 1),
    num_classes=NUM_CLASSES,
)

model.summary()


# ============================================================
# CTC LOSS
# ============================================================

def ctc_loss(y_true, y_pred):

    batch_size = tf.shape(y_true)[0]

    input_length = tf.shape(y_pred)[1]

    label_length = tf.shape(y_true)[1]

    input_length = tf.cast(
        input_length,
        dtype=tf.int64
    )

    label_length = tf.cast(
        label_length,
        dtype=tf.int64
    )

    input_length = tf.ones(
        shape=(batch_size, 1),
        dtype=tf.int64
    ) * input_length

    label_length = tf.ones(
        shape=(batch_size, 1),
        dtype=tf.int64
    ) * label_length

    return tf.keras.backend.ctc_batch_cost(
        y_true,
        y_pred,
        input_length,
        label_length,
    )


# ============================================================
# COMPILE
# ============================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0005,
    clipnorm=1.0,
)

model.compile(
    optimizer=optimizer,
    loss=ctc_loss,
)


# ============================================================
# CALLBACKS
# ============================================================

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(BEST_MODEL_PATH),
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1,
)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1,
)


# ============================================================
# TRAIN
# ============================================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[
        checkpoint_cb,
        early_stopping_cb,
    ],
)


# ============================================================
# SAVE
# ============================================================

model.save(
    MODEL_DIR / "best_crnn_model.keras"
)

print(
    f"Best model saved to:\n"
    f"{BEST_MODEL_PATH}"
)