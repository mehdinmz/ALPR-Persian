import tensorflow as tf
from model import build_crnn_model
import argparse
import sys
from pathlib import Path
from Ldataset import get_dataset, VOCABULARY
DATA_DIR = Path("../data").resolve()

model = build_crnn_model(input_shape=(32, 128, 1), num_classes=45)
model.summary()


# 1. دریافت دیتاست‌ها
train_ds, val_ds = get_dataset("../data/synthetic_plates_crnn", batch_size=64)

# 2. ساخت مدل (ابعاد تصویر 32x160 و تعداد کلاس‌ها برابر با طول Vocabulary + 1)
num_classes = len(VOCABULARY) + 2  # تعداد کل کاراکترها + OOV + Blank token در CTC
model = build_crnn_model(input_shape=(32, 160, 1), num_classes=num_classes)

# 3. کامپایل مدل با CTC Loss
def ctc_loss_lambda(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
def train():
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath="best_crnn_model.keras", # ذخیره مدل کامل با فرمت جدید Keras
        monitor="val_loss",
        save_best_only=True,               # فقط زمانی که val_loss بهتر شد ذخیره کن
        mode="min",
        verbose=0
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0005,
        clipnorm=1.0  # محدود کردن سقف گرادیان‌ها
    )
    model.compile(optimizer=optimizer, loss=ctc_loss_lambda)

    # 4. اجرا
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=25,
        callbacks=[checkpoint_cb] # اضافه کردن Callback به fit
    )

    # 5. دستی ذخیره کردن مدل در انتهای آموزش (در صورت نیاز)
    model.save("best_crnn_model.keras")
    print("مدل با موفقیت ذخیره شد!")