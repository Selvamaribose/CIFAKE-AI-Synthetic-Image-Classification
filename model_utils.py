from pathlib import Path

from tensorflow import keras


DEFAULT_IMG_SIZE = 160
LEGACY_IMG_SIZE = 48
ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "AIGeneratedModel.weights.h5"
LEGACY_MODEL_PATH = ROOT_DIR / "AIGeneratedModel.h5"
METRICS_PATH = ROOT_DIR / "training_metrics.json"


def create_transfer_learning_model(
    img_size=DEFAULT_IMG_SIZE,
    pretrained=False,
    dropout_rate=0.2,
    augment=False,
):
    weights = "imagenet" if pretrained else None
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = inputs

    if augment:
        x = keras.Sequential(
            [
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(0.08),
                keras.layers.RandomZoom(0.1),
                keras.layers.RandomContrast(0.1),
            ],
            name="augmentation",
        )(x)

    x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(x)

    base_model = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights=weights,
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = keras.Model(inputs, outputs, name="ai_image_classifier")
    return model, base_model


def build_classifier_model(img_size=DEFAULT_IMG_SIZE, pretrained=False):
    model, _ = create_transfer_learning_model(
        img_size=img_size,
        pretrained=pretrained,
        augment=False,
    )
    return model


def get_model_image_size(model):
    input_shape = getattr(model, "input_shape", None)
    if not input_shape or len(input_shape) < 3 or input_shape[1] is None:
        return DEFAULT_IMG_SIZE
    return int(input_shape[1])
