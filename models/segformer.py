import tensorflow as tf
from tensorflow.keras import layers
from .modules import MixVisionTransformer
from .Head import SegFormerHead
from .utils import ResizeLayer

MODEL_CONFIGS = {
    "mit_b0": {
        "embed_dims": [32, 64, 160, 256],
        "depths": [2, 2, 2, 2],
        "decode_dim": 256,
    },
    "mit_b1": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [2, 2, 2, 2],
        "decode_dim": 256,
    },
    "mit_b2": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [3, 4, 6, 3],
        "decode_dim": 768,
    },
    "mit_b3": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [3, 4, 18, 3],
        "decode_dim": 768,
    },
    "mit_b4": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [3, 8, 27, 3],
        "decode_dim": 768,
    },
    "mit_b5": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [3, 6, 40, 3],
        "decode_dim": 768,
    },
}


def SegFormer_B0(input_shape, num_classes):
    pixel_values = layers.Input(shape=input_shape, name="pixel_values")
    x = MixVisionTransformer(
        img_size=input_shape[1],
        embed_dims=MODEL_CONFIGS["mit_b0"]["embed_dims"],
        depths=MODEL_CONFIGS["mit_b0"]["depths"],
    )(pixel_values)
    x = layers.LayerNormalization()(x)
    x = SegFormerHead(
        num_classes=num_classes,
        decode_dim=MODEL_CONFIGS["mit_b0"]["decode_dim"],
    )(x)

    x = ResizeLayer(input_shape[0], input_shape[1])(x)
    x = tf.keras.activations.softmax(x,)
    return tf.keras.Model(inputs=pixel_values, outputs=x)


def SegFormer_B1(input_shape, num_classes):
    pixel_values = layers.Input(shape=input_shape, name="pixel_values")
    x = MixVisionTransformer(
        img_size=input_shape[1],
        embed_dims=MODEL_CONFIGS["mit_b1"]["embed_dims"],
        depths=MODEL_CONFIGS["mit_b1"]["depths"],
    )(pixel_values)
    x = layers.LayerNormalization()(x)
    x = SegFormerHead(
        num_classes=num_classes,
        decode_dim=MODEL_CONFIGS["mit_b1"]["decode_dim"],
    )(x)

    x = ResizeLayer(input_shape[0], input_shape[1])(x)
    x = tf.keras.activations.softmax(x,)
    return tf.keras.Model(inputs=pixel_values, outputs=x)


def SegFormer_B2(input_shape, num_classes):
    pixel_values = layers.Input(shape=input_shape, name="pixel_values")
    x = MixVisionTransformer(
        img_size=input_shape[1],
        embed_dims=MODEL_CONFIGS["mit_b2"]["embed_dims"],
        depths=MODEL_CONFIGS["mit_b2"]["depths"],
    )(pixel_values)
    x = layers.LayerNormalization()(x)
    x = SegFormerHead(
        num_classes=num_classes,
        decode_dim=MODEL_CONFIGS["mit_b2"]["decode_dim"],
    )(x)

    x = ResizeLayer(input_shape[0], input_shape[1])(x)
    x = tf.keras.activations.softmax(x,)
    return tf.keras.Model(inputs=pixel_values, outputs=x)


def SegFormer_B3(input_shape, num_classes):
    pixel_values = layers.Input(shape=input_shape, name="pixel_values")
    x = MixVisionTransformer(
        img_size=input_shape[1],
        embed_dims=MODEL_CONFIGS["mit_b3"]["embed_dims"],
        depths=MODEL_CONFIGS["mit_b3"]["depths"],
    )(pixel_values)
    x = layers.LayerNormalization()(x)
    x = SegFormerHead(
        num_classes=num_classes,
        decode_dim=MODEL_CONFIGS["mit_b3"]["decode_dim"],
    )(x)

    x = ResizeLayer(input_shape[0], input_shape[1])(x)
    x = tf.keras.activations.softmax(x,)
    return tf.keras.Model(inputs=pixel_values, outputs=x)


def SegFormer_B4(input_shape, num_classes):
    pixel_values = layers.Input(shape=input_shape, name="pixel_values")
    x = MixVisionTransformer(
        img_size=input_shape[1],
        embed_dims=MODEL_CONFIGS["mit_b4"]["embed_dims"],
        depths=MODEL_CONFIGS["mit_b4"]["depths"],
    )(pixel_values)
    x = layers.LayerNormalization()(x)
    x = SegFormerHead(
        num_classes=num_classes,
        decode_dim=MODEL_CONFIGS["mit_b4"]["decode_dim"],
    )(x)

    x = ResizeLayer(input_shape[0], input_shape[1])(x)
    x = tf.keras.activations.softmax(x,)
    return tf.keras.Model(inputs=pixel_values, outputs=x)


def SegFormer_B5(input_shape, num_classes):
    pixel_values = layers.Input(shape=input_shape, name="pixel_values")
    x = MixVisionTransformer(
        img_size=input_shape[1],
        embed_dims=MODEL_CONFIGS["mit_b5"]["embed_dims"],
        depths=MODEL_CONFIGS["mit_b5"]["depths"],
    )(pixel_values)
    x = layers.LayerNormalization()(x)
    x = SegFormerHead(
        num_classes=num_classes,
        decode_dim=MODEL_CONFIGS["mit_b5"]["decode_dim"],
    )(x)

    x = ResizeLayer(input_shape[0], input_shape[1])(x)
    x = tf.keras.activations.softmax(x,)
    return tf.keras.Model(inputs=pixel_values, outputs=x)
