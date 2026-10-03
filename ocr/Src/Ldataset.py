from pathlib import Path

import numpy as np
import tensorflow as tf


# ============================================================
# VOCABULARY
# ============================================================

DIGITS = [
    "0", "1", "2", "3", "4",
    "5", "6", "7", "8", "9"
]

LETTERS = [
    "alef",
    "beh",
    "peh",
    "teh",
    "seh",
    "jim",
    "cheh",
    "heh",
    "kheh",
    "dal",
    "zal",
    "reh",
    "zeh",
    "zheh",
    "sin",
    "shin",
    "sad",
    "zad",
    "tah",
    "zah",
    "ein",
    "ghein",
    "feh",
    "ghaf",
    "kaf",
    "gaf",
    "lam",
    "mim",
    "noon",
    "vav",
    "he",
    "ye",
]

VOCABULARY = DIGITS + LETTERS

TOKEN_TO_ID = {
    token: index + 1
    for index, token in enumerate(VOCABULARY)
}

ID_TO_TOKEN = {
    index + 1: token
    for index, token in enumerate(VOCABULARY)
}

MAX_LABEL_LEN = 8


# ============================================================
# LABEL ENCODING
# ============================================================

def encode_label(label_text: str):
    """
    Convert folder name into 8 OCR tokens.

    Example:

        19teh933_20

    becomes:

        [
            "1",
            "9",
            "teh",
            "9",
            "3",
            "3",
            "2",
            "0"
        ]

    The letter token is treated as ONE token.
    """

    label_text = label_text.strip().lower()

    # --------------------------------------------------------
    # Split city code
    # --------------------------------------------------------

    parts = label_text.split("_")

    if len(parts) != 2:
        raise ValueError(
            f"Invalid plate folder name: {label_text!r}. "
            f"Expected format like: 19teh933_20"
        )

    main_part = parts[0]
    city_code = parts[1]

    # --------------------------------------------------------
    # Validate city code
    # --------------------------------------------------------

    if len(city_code) != 2 or not city_code.isdigit():
        raise ValueError(
            f"Invalid city code in plate: {label_text!r}"
        )

    # --------------------------------------------------------
    # First two digits
    # --------------------------------------------------------

    if len(main_part) < 6:
        raise ValueError(
            f"Invalid main plate section: {label_text!r}"
        )

    first_two = main_part[:2]

    if not first_two.isdigit():
        raise ValueError(
            f"First two characters must be digits: "
            f"{label_text!r}"
        )

    # --------------------------------------------------------
    # Extract letter token
    #
    # Example:
    #
    #     19teh933
    #       ^^
    #
    #     Actually "teh" is detected as the alphabetic
    #     sequence between the first 2 digits and final 3 digits.
    # --------------------------------------------------------

    remaining = main_part[2:]

    letter_end = 0

    while (
        letter_end < len(remaining)
        and remaining[letter_end].isalpha()
    ):
        letter_end += 1

    letter = remaining[:letter_end]
    middle_three = remaining[letter_end:]

    # --------------------------------------------------------
    # Validate letter
    # --------------------------------------------------------

    if not letter:
        raise ValueError(
            f"Letter token not found in plate: "
            f"{label_text!r}"
        )

    if letter not in LETTERS:
        raise ValueError(
            f"Unknown letter token {letter!r} "
            f"in plate {label_text!r}"
        )

    # --------------------------------------------------------
    # Validate middle three digits
    # --------------------------------------------------------

    if len(middle_three) != 3:
        raise ValueError(
            f"Expected 3 digits after letter in "
            f"{label_text!r}, got {middle_three!r}"
        )

    if not middle_three.isdigit():
        raise ValueError(
            f"Middle section must contain 3 digits: "
            f"{label_text!r}"
        )

    # --------------------------------------------------------
    # Build 8 tokens
    # --------------------------------------------------------

    tokens = (
        list(first_two)
        + [letter]
        + list(middle_three)
        + list(city_code)
    )

    # --------------------------------------------------------
    # Final length check
    # --------------------------------------------------------

    if len(tokens) != MAX_LABEL_LEN:
        raise ValueError(
            f"Expected {MAX_LABEL_LEN} tokens, "
            f"got {len(tokens)} for plate "
            f"{label_text!r}"
        )

    # --------------------------------------------------------
    # Convert tokens to integer IDs
    # --------------------------------------------------------

    encoded = []

    for token in tokens:

        if token not in TOKEN_TO_ID:
            raise ValueError(
                f"Unknown token {token!r} "
                f"in plate {label_text!r}"
            )

        encoded.append(
            TOKEN_TO_ID[token]
        )

    return np.asarray(
        encoded,
        dtype=np.int32
    )


# ============================================================
# LOAD ALL PLATES
# ============================================================

def load_dataset_from_folders(data_dir):
    """
    Expected structure:

        data_dir/
        ├── 19teh933_20/
        │   ├── 0.png
        │   ├── 1.png
        │   └── ...
        │
        ├── 52sin123_45/
        │   ├── 0.png
        │   └── ...
        │
        └── ...

    The folder name is the label.

    Returns:

        image_paths
        labels
        plate_ids
    """

    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n"
            f"{data_dir.resolve()}"
        )

    plate_dirs = sorted(
        [
            path
            for path in data_dir.iterdir()
            if path.is_dir()
        ]
    )

    if not plate_dirs:
        raise RuntimeError(
            f"No plate directories found in:\n"
            f"{data_dir.resolve()}"
        )

    image_paths = []
    labels = []
    plate_ids = []

    # --------------------------------------------------------
    # Process each plate folder
    # --------------------------------------------------------

    for plate_dir in plate_dirs:

        plate_id = plate_dir.name

        # Encode folder name ONCE
        encoded_label = encode_label(
            plate_id
        )

        # ----------------------------------------------------
        # All PNG files belong to this plate
        # ----------------------------------------------------

        images = sorted(
            plate_dir.glob("*.png"),
            key=lambda path: (
                int(path.stem)
                if path.stem.isdigit()
                else path.stem
            )
        )

        if not images:
            print(
                f"Warning: no PNG images found in "
                f"{plate_dir}"
            )
            continue

        for image_path in images:

            image_paths.append(
                str(image_path)
            )

            labels.append(
                encoded_label
            )

            plate_ids.append(
                plate_id
            )

    if not image_paths:
        raise RuntimeError(
            f"No images found in:\n"
            f"{data_dir.resolve()}"
        )

    return (
        np.asarray(image_paths),
        np.asarray(labels, dtype=np.int32),
        np.asarray(plate_ids)
    )


# ============================================================
# IMAGE PROCESSING
# ============================================================

def process_sample(image_path, label):

    image = tf.io.read_file(
        image_path
    )

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

    # --------------------------------------------------------
    # Default path
    # --------------------------------------------------------

    if data_dir is None:

        data_dir = (
            Path(__file__).resolve().parent.parent
            / "Dataset"
            / "generated_plates"
            / "roya_bold"
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    (
        image_paths,
        labels,
        plate_ids
    ) = load_dataset_from_folders(
        data_dir
    )

    # --------------------------------------------------------
    # Split by UNIQUE PLATE
    #
    # IMPORTANT:
    # Do NOT split individual images.
    # All variations of one plate must stay
    # in the same split.
    # --------------------------------------------------------

    unique_plates = np.unique(
        plate_ids
    )

    num_plates = len(
        unique_plates
    )

    if num_plates < 2:
        raise RuntimeError(
            "At least 2 unique plates are required."
        )

    rng = np.random.default_rng(
        seed
    )

    shuffled_plates = unique_plates.copy()

    rng.shuffle(
        shuffled_plates
    )

    val_plate_count = max(
        1,
        int(
            num_plates
            * validation_split
        )
    )

    val_plate_ids = set(
        shuffled_plates[
            :val_plate_count
        ]
    )

    train_mask = np.array(
        [
            plate_id not in val_plate_ids
            for plate_id in plate_ids
        ],
        dtype=bool
    )

    val_mask = np.array(
        [
            plate_id in val_plate_ids
            for plate_id in plate_ids
        ],
        dtype=bool
    )

    train_paths = image_paths[
        train_mask
    ]

    train_labels = labels[
        train_mask
    ]

    val_paths = image_paths[
        val_mask
    ]

    val_labels = labels[
        val_mask
    ]

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("=" * 60)
    print("OCR DATASET")
    print("=" * 60)

    print(
        f"Dataset : {data_dir.resolve()}"
    )

    print(
        f"Unique plates : {num_plates}"
    )

    print(
        f"Total images : {len(image_paths)}"
    )

    print()

    print(
        f"Train plates : "
        f"{num_plates - val_plate_count}"
    )

    print(
        f"Validation plates : "
        f"{val_plate_count}"
    )

    print()

    print(
        f"Train images : "
        f"{len(train_paths)}"
    )

    print(
        f"Validation images : "
        f"{len(val_paths)}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # TensorFlow Dataset
    # --------------------------------------------------------

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                train_paths,
                train_labels
            )
        )
        .shuffle(
            buffer_size=min(
                len(train_paths),
                10000
            ),
            seed=seed,
            reshuffle_each_iteration=True
        )
        .map(
            process_sample,
            num_parallel_calls=AUTOTUNE
        )
        .batch(
            batch_size
        )
        .prefetch(
            AUTOTUNE
        )
    )

    val_ds = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                val_paths,
                val_labels
            )
        )
        .map(
            process_sample,
            num_parallel_calls=AUTOTUNE
        )
        .batch(
            batch_size
        )
        .prefetch(
            AUTOTUNE
        )
    )

    return train_ds, val_ds