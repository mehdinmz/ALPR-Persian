import tensorflow as tf
from tensorflow.keras import layers, models

def build_crnn_model(
    input_shape=(32, 128, 1), # عرض تصویر معمولاً برای توالی‌ها بزرگ‌تر از ارتفاع است
    num_classes=45            # تعداد کلاس‌ها (شامل کاراکترها + علامت blank برای CTC)
):
    inputs = layers.Input(shape=input_shape, name="image_input")

    # ==========================================
    # 1. CNN Feature Extractor
    # ==========================================
    # Block 1
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="pool1")(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="pool2")(x)

    # Block 3 & 4
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)

    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4")(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name="bn4")(x)
    x = layers.ReLU(name="relu4")(x)
    x = layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name="pool3")(x)

    # Block 5 & 6
    x = layers.Conv2D(512, (3, 3), padding="same", name="conv5")(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name="bn5")(x)
    x = layers.ReLU(name="relu5")(x)

    x = layers.Conv2D(512, (3, 3), padding="same", name="conv6")(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name="bn6")(x)
    x = layers.ReLU(name="relu6")(x)
    x = layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name="pool4")(x)

    # Block 7
    x = layers.Conv2D(512, (3, 3), padding="same", name="conv7")(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name="bn7")(x)
    x = layers.ReLU(name="relu7")(x)

    # ==========================================
    # 2. Map to Sequence
    # ==========================================
    # ورودی در این مرحله ابعادی معادل (Batch, Height, Width, Channels) دارد.
    # ارتفاع به ۱ کاهش یافته (در ورودی H=32: 32 -> 16 -> 8 -> 4 -> 2 -> 1)
    # خروجی Reshape به صورت (Batch, Width, Height * Channels) در می‌آید.
    
    # تغییر ابعاد: تبدیل ارتفاع و کانال‌ها به یک بُعد ویژگی
    target_shape = (-1, x.shape[2], x.shape[1] * x.shape[3]) # (TimeSteps, Features)
    x = layers.Permute((2, 1, 3))(x) # جا به جایی عرض و ارتفاع جهت همخوانی با گام زمان
    x = layers.Reshape((-1, x.shape[2] * x.shape[3]), name="reshape")(x)

    # معادل لایه Linear(1024, 96) در مدل PyTorch
    x = layers.Dense(96, activation="relu", name="map2seq")(x)

    # ==========================================
    # 3. Recurrent Layers (BiLSTM)
    # ==========================================
    # rnn1: LSTM(96, 256, bidirectional=True)
    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True), name="rnn1"
    )(x)

    # rnn2: LSTM(512, 256, bidirectional=True)
    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True), name="rnn2"
    )(x)

    # ==========================================
    # 4. Classifier
    # ==========================================
    # خروجی احتمالات برای هر گام زمانی (Time Step)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CRNN_Image2Text")
    return model