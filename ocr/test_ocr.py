import numpy as np
import tensorflow as tf
from PIL import Image

# =========================
# PATHS
# =========================

IMAGE_PATH = r"D:\Projects\ALPR-Persian\ocr\Dataset\generated_plates\roya_bold\48feh766_55\0.png"
MODEL_PATH = r"Models\best_crnn_model.keras"

# =========================
# VOCABULARY
# =========================

DIGITS = [
    "0", "1", "2", "3", "4",
    "5", "6", "7", "8", "9"
]

LETTERS = [
    "D",
    "S",
    "alef",
    "beh",
    "dal",
    "ein",
    "feh",
    "gaf",
    "ghaf",
    "he",
    "jim",
    "kaf",
    "lam",
    "mim",
    "noon",
    "peh",
    "sad",
    "seh",
    "shin",
    "sin",
    "ta",
    "teh",
    "vav",
    "ye",
    "zeh",
    "zhe",
]

VOCABULARY = DIGITS + LETTERS

ID_TO_TOKEN = {
    index + 1: token
    for index, token in enumerate(VOCABULARY)
}

BLANK_ID = 0
UNKNOWN_ID = len(VOCABULARY) + 1

# =========================
# LOAD MODEL
# =========================

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

print("Input :", model.input_shape)
print("Output:", model.output_shape)

# =========================
# LOAD IMAGE
# =========================

image = Image.open(IMAGE_PATH).convert("L")

print("Original image:", image.size)

image = image.resize(
    (160, 32),
    Image.Resampling.BILINEAR,
)

image = np.asarray(
    image,
    dtype=np.float32,
)

image /= 255.0

image = np.expand_dims(image, axis=-1)
image = np.expand_dims(image, axis=0)

print("Input tensor:", image.shape)
print("Min:", image.min())
print("Max:", image.max())

# =========================
# PREDICTION
# =========================

prediction = model.predict(
    image,
    verbose=0,
)

print("Prediction shape:", prediction.shape)

# =========================
# RAW ARGMAX
# =========================

ids = np.argmax(
    prediction[0],
    axis=-1,
)

print("\nRAW ARGMAX:")
print(ids.tolist())

# =========================
# RAW TOKENS
# =========================

print("\nRAW TOKENS:")

for i, class_id in enumerate(ids):

    token = ID_TO_TOKEN.get(
        int(class_id),
        "<BLANK/UNKNOWN>"
    )

    print(
        f"{i:02d}: "
        f"class={class_id:02d} "
        f"token={token}"
    )

# =========================
# CTC DECODE
# =========================

input_length = np.array(
    [prediction.shape[1]],
    dtype=np.int32,
)

decoded, _ = tf.keras.backend.ctc_decode(
    prediction,
    input_length=input_length,
    greedy=True,
)

decoded_ids = decoded[0].numpy()[0]

print("\nCTC DECODE IDS:")
print(decoded_ids.tolist())

result = []

for class_id in decoded_ids:

    class_id = int(class_id)

    if class_id < 0:
        continue

    if class_id == BLANK_ID:
        continue

    if class_id == UNKNOWN_ID:
        result.append("?")
        continue

    token = ID_TO_TOKEN.get(class_id)

    if token is not None:
        result.append(token)

print("\nFINAL RESULT:")
print("".join(result))