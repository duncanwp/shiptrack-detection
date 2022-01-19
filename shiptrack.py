#!/usr/bin/env python
import argparse, os
import numpy as np
from pathlib import Path

import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# c.f. https://github.com/qubvel/segmentation_models/issues/374
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import *
from segmentation_models.metrics import iou_score

# Tensorflow needs image channels last, e.g. (batch size, width, height, channels)
K.set_image_data_format('channels_last')
print(K.image_data_format())

losses = {'jaccard_loss': jaccard_loss,
          'dice_loss': dice_loss,
          'binary_focal_loss': binary_focal_loss,
          'binary_crossentropy': binary_crossentropy,
          'bce_dice_loss': bce_dice_loss,
          'bce_jaccard_loss': bce_jaccard_loss,
          'binary_focal_dice_loss': binary_focal_dice_loss,
          'binary_focal_jaccard_loss': binary_focal_jaccard_loss}

# Set default values. These need to be global because the tf.Dataset.map used for pp above doesn't accept kwargs
IMG_SIZE = 448
INT_IMG_SIZE = (2240, 1344)
AUTOTUNE = tf.data.AUTOTUNE  #args.pop('autotune')


def fit_model(model, train_dataset, val_dataset, epochs=1, tensorboard_dir=''):

    tensorboard = TensorBoard(log_dir=tensorboard_dir, histogram_freq=0,
                              write_images=True, write_graph=False)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=5e-7, verbose=1)

    history = model.fit(train_dataset, validation_data=val_dataset, verbose=1,
                        epochs=epochs, callbacks=[tensorboard, reduce_lr])

    return history


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_numpy_arrays(training_dir):
    import os

    all_data = np.memmap(os.path.join(training_dir, 'data.memmap'), dtype='uint8', mode='r', shape=(2205, 2030, 1354, 4))
    all_labels = np.memmap(os.path.join(training_dir, 'labels.memmap'), dtype='uint8', mode='r', shape=(2205, 2030, 1354))

    # Drop the alpha
    if all_data.shape[3] == 4:
        all_data = all_data[..., :3]

    return all_data, all_labels

def get_generator(all_data, all_labels):

    for data, labels in zip(all_data, all_labels):
        # Resize the data
        _data = tf.image.resize(data, INT_IMG_SIZE) / 255.
        _labels = tf.image.resize(tf.expand_dims(labels, -1), INT_IMG_SIZE, 'nearest') # Adding an extra color dim for tf.image

        # Slice the images to the final size...
        flat_patches = tf.image.extract_patches(images=tf.expand_dims(_data, axis=0),
                                                sizes=[1, IMG_SIZE, IMG_SIZE, 1],
                                                strides=[1, IMG_SIZE, IMG_SIZE, 1],  # This should be the same as sizes
                                                rates=[1, 1, 1, 1],
                                                padding='VALID')
        _data = tf.reshape(flat_patches, [-1, IMG_SIZE, IMG_SIZE, 3])  # Stack them along the leading dim

        # ...And the labels
        flat_patches = tf.image.extract_patches(images=tf.expand_dims(_labels, axis=0),
                                                sizes=[1, IMG_SIZE, IMG_SIZE, 1],
                                                strides=[1, IMG_SIZE, IMG_SIZE, 1],  # This should be the same as sizes
                                                rates=[1, 1, 1, 1],
                                                padding='VALID')
        _labels = tf.reshape(flat_patches, [-1, IMG_SIZE, IMG_SIZE])  # Stack them along the leading dim

        has_labels = tf.math.reduce_any(tf.reshape(_labels, [-1, IMG_SIZE*IMG_SIZE]) > 0, axis=1)
        _data = tf.boolean_mask(_data, has_labels)
        _labels = tf.boolean_mask(_labels, has_labels)
        yield _data, _labels

        
def resize_and_rescale(image, label):
    image = tf.cast(image, tf.float32)

    # Resize to the intermediate size (which might include a downscale)
    image = tf.image.resize(image, INT_IMG_SIZE)
    label = tf.squeeze(tf.image.resize(tf.expand_dims(label, -1), INT_IMG_SIZE, 'nearest')) # Adding an extra color dim for tf.image
    image = (image / 255.0)

    return image, label


def augment_images(image_label, seed):
    image, label = image_label
    image, label = resize_and_rescale(image, label)
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + IMG_SIZE//20, IMG_SIZE + IMG_SIZE//20)
    label = tf.image.resize_with_crop_or_pad(label, IMG_SIZE + IMG_SIZE // 20, IMG_SIZE + IMG_SIZE // 20)
    # Make a new seed
    new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
    # Random crop back to the original size
    image = tf.image.stateless_random_crop(
      image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
    label = tf.image.stateless_random_crop(
      label, size=[IMG_SIZE, IMG_SIZE], seed=seed)
    # Random brightness
    image = tf.image.stateless_random_brightness(
      image, max_delta=0.5, seed=new_seed)  # (not the label for this one)
    # Random flip
    image = tf.image.stateless_random_flip_left_right(
      image, seed=new_seed)
    label = tf.image.stateless_random_flip_up_down(
      label, seed=new_seed)
    image = tf.clip_by_value(image, 0, 1)
    return image, label


def get_data(training_dir, test_prop, batch_size, augment):

    all_data, all_labels = load_numpy_arrays(training_dir)

    n_test = (all_data.shape[0] // 100) * test_prop
    n_val = int((all_data.shape[0]-n_test)*0.181818)  # Fixed validation proportion of ~15% of original dataset
    n_train = all_data.shape[0]-n_test-n_val
    
    x_test, x_val, x_train = all_data[:n_test], all_data[n_test:n_test+n_val], all_data[n_test+n_val:]
    y_test, y_val, y_train = all_labels[:n_test], all_labels[n_test:n_test+n_val], all_labels[n_test+n_val:]

    print(n_test, 'test samples')
    print(n_val, 'val samples')
    print(n_train, 'train samples')

    sig = output_signature=(
         tf.TensorSpec(shape=(15, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float64),
         tf.TensorSpec(shape=(15, IMG_SIZE, IMG_SIZE), dtype=tf.int32))
    test_ds = tf.data.Dataset.from_generator(get_generator, args=(x_test, y_test), output_signature=sig).unbatch()

    val_ds = tf.data.Dataset.from_generator(get_generator, args=(x_val, y_val), output_signature=sig).unbatch()

    train_ds = tf.data.Dataset.from_generator(get_generator, args=(x_train, y_train), output_signature=sig).unbatch()


    if augment:
        # Create counter and zip together with train dataset
        counter = tf.data.experimental.Counter()
        train_ds = tf.data.Dataset.zip((train_ds, (counter, counter))).map(augment_images, num_parallel_calls=AUTOTUNE)
#        train_fn = augment_images
#    else:
#        train_fn = resize_and_rescale

    # TODO: Apply this fn to each of the below datasets *after* the map so it applies on the individual slices
    #   I don't think it's nans I need to worry about though and it's not clear how much of a problem it is so
    #    it's not plumbed in yet
    def no_missing_vals(x, y):
        return not tf.manth.any(tf.math.is_nan(x))

    train_ds = (
        train_ds
#             .shuffle(10)
#            .map(train_fn, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
#             .prefetch(AUTOTUNE)
    )

    val_ds = (
        val_ds
#            .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
#             .prefetch(AUTOTUNE)
    )

    test_ds = (
        test_ds
#            .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
#             .prefetch(AUTOTUNE)
    )

    return test_ds, val_ds, train_ds


def main(training, epochs, learning_rate, batch_size, model_dir,
         backbone, augment, loss, encoder_freeze, test_prop, tensorboard_dir):

    test_ds, val_ds, train_ds = get_data(training, test_prop, batch_size, augment)

    print("got data")

    # preprocess input
    # This dies when evaluating or iterating on the x_train Dataset with the following exception:
    # [libprotobuf FATAL external/com_google_protobuf/src/google/protobuf/wire_format_lite.cc:504] CHECK failed: (value.size()) <= (kint32max): 
    # Segmentation fault (core dumped)

    # The problem seems to be related to either a memory leak in the from_generator method or this: https://github.com/tensorflow/tensorflow/issues/46089
    print(list(train_ds.take(3)))
    
    print("processed input")
    
    # Automatically mirror training across all available GPUs
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        # define model
        model = Unet(backbone, encoder_weights='imagenet', encoder_freeze=encoder_freeze,
                     classes=1, activation='sigmoid')

        print(model.summary())

        model.compile(Adam(lr=learning_rate), loss=losses[loss], metrics=[iou_score])

    history = fit_model(model, train_ds, val_ds, epochs=epochs, tensorboard_dir=tensorboard_dir)

    score = model.evaluate(test_ds, verbose=0)

    print('Test loss    :', score[0])
    print('Test accuracy:', score[1])

    # save Keras model for Tensorflow Serving
    tf.saved_model.save(
        model,
        os.path.join(model_dir, 'model/1'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', ''))
    parser.add_argument('--tensorboard-dir', type=str, default=Path(os.environ.get('SM_MODULE_DIR', '')).parent.parent)
    parser.add_argument('--training', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
#     parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument("--augment", type=str2bool, nargs='?',
                            const=True, default=False,
                            help="Augment the training data.")
#    parser.add_argument("--autotune", type=str2bool, nargs='?',
#                            const=True, default=False,
#                            help="Allow tf to autotune parallel processing.")
    parser.add_argument("--encoder-freeze", type=str2bool, nargs='?',
                            const=True, default=False,
                            help="Freeze the weights of the encoding layer.")
    parser.add_argument('--backbone', default='resnet152')
    parser.add_argument('--loss', default='bce_jaccard_loss', choices=list(losses.keys()))
    parser.add_argument('--test-prop', type=int, default=10,
                        help="Percentage of images to use for testing")
    parser.add_argument('--image-size', type=int, default=448,
                        help="Image size (in pixels) for training")
    parser.add_argument('--intermediate-image-size', nargs=2, default=(2240, 1344),
                        help="Intermediate image size (in pixels) before slicing to final image-size. "
                             "Can include a downsample. The default (2240, 1344) corresponds to an approximately "
                             "5x3 image split for a final size of 448px")

    args = vars(parser.parse_args())

    # These need to be global because the tf.Dataset.map used for pp above doesn't accept kwargs
    IMG_SIZE = args.pop('image_size')
    INT_IMG_SIZE = args.pop('intermediate_image_size')
    AUTOTUNE = tf.data.AUTOTUNE  #args.pop('autotune')

    main(**args)
