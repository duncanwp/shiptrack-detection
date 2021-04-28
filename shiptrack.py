import argparse, os
import numpy as np
from pathlib import Path

import tensorflow as tf

from keras import backend as K
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

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


def get_data_flow(data, labels, subset, batch_size=1):
    # this is the augmentation configuration we will use for training
    from keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2,
        fill_mode='constant')

    generator = datagen.flow(
        data, y=labels,
        batch_size=batch_size if subset == 'training' else 1,
        subset=subset)

    return generator


def fit_model(model, training_data, labels, batch_size=1, epochs=1, augment=True, tensorboard_dir=''):

    tensorboard = TensorBoard(log_dir=tensorboard_dir, histogram_freq=0,
                              write_images=True, write_graph=False)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=5e-7, verbose=1)

    if augment:
        train_generator = get_data_flow(training_data, labels, subset='training')
        validation_generator = get_data_flow(training_data, labels, ubset='validation')

        history = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=len(train_generator),
                                      epochs=epochs,
                                      validation_steps=1,
                                      validation_data=validation_generator,
                                      callbacks=[tensorboard, reduce_lr],
                                      # use_multiprocessing=True, workers=8,
                                      verbose=1)
    else:
        ds = tf.data.Dataset.from_tensor_slices((training_data, labels))
        n_val = int(training_data.shape[0]*0.181818)
        n_test = int(training_data.shape[0]) - n_val
        print(n_val)
        val_dataset = ds.take(n_val).batch(batch_size)
        train_dataset = ds.skip(n_val).shuffle(n_test, reshuffle_each_iteration=True).batch(batch_size)
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


def get_data(training_dir, test_prop, channel=None, downsample=False):
    import os
    
    if os.path.isfile(os.path.join(training_dir, 'data.npz')):
        all_data = np.load(os.path.join(training_dir, 'data.npz'))['arr_0']
    elif os.path.isfile(os.path.join(training_dir, 'data.npy')):
        all_data = np.load(os.path.join(training_dir, 'data.npy'))
    else:
        raise ValueError("No training data found")
        
    all_labels = np.load(os.path.join(training_dir, 'labels.npz'))['arr_0']

    # If we have more than one class, collapse them all together (we're not trying different types)
    if all_labels.shape[3] > 2:
        all_labels[..., 1] = all_labels.any(axis=3).astype('uint8')
        all_labels = all_labels[..., 0:2]
    
    scaled_data = all_data.astype('float32') / 255.
    if channel is not None:
        scaled_data = scaled_data[..., channel]
    else:
        # Drop the alpha
        scaled_data = scaled_data[..., :3]

    padded_data = np.pad(scaled_data, ((0, 0), (4, 4), (4, 4), (0, 0)), mode='mean')
    # Pads with zeros by default
    padded_labels = np.pad(all_labels[..., 1:2], ((0, 0), (4, 4), (4, 4), (0, 0)), mode='constant')
    
    # Normalise the data, we can use a constant mean and std calculated across all the imagery
    #     This shouldn't be needed for sigmoid activation
#     data_norm = (padded_data - 0.55) / 0.15
    data_norm = padded_data
    
    if downsample:
        new_shape = [data_norm.shape[1]//2, data_norm.shape[2]//2]
        data_norm = tf.image.resize(data_norm, new_shape, antialias=True).numpy()
        padded_labels = tf.image.resize(padded_labels, new_shape, method='nearest').numpy()

    n_test = (data_norm.shape[0] // 100) * test_prop
    x_test, y_test = data_norm[:n_test, ...], padded_labels[:n_test, ...]
    x_train, y_train = data_norm[n_test:, ...], padded_labels[n_test:, ...]

    print(x_train.shape[0], 'train/val samples')
    print(x_test.shape[0], 'test samples')

    return x_test, x_train, y_test, y_train


def main(training, epochs, learning_rate, batch_size, model_dir, channel,
         backbone, augment, loss, encoder_freeze, test_prop, tensorboard_dir, downsample):

    x_test, x_train, y_test, y_train = get_data(training, test_prop, channel, downsample)

    # preprocess input
    preprocess_input = get_preprocessing(backbone)
    x_train = preprocess_input(x_train)
    
    # Automatically mirror training across all available GPUs
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        # define model
        model = Unet(backbone, encoder_weights='imagenet', encoder_freeze=encoder_freeze, classes=1,
                     activation='sigmoid')

        print(model.summary())

        model.compile(Adam(lr=learning_rate), loss=losses[loss], metrics=[iou_score])

    history = fit_model(model, x_train, y_train, batch_size=batch_size,
                        epochs=epochs, augment=augment, tensorboard_dir=tensorboard_dir)

    score = model.evaluate(x_test, y_test, verbose=0)

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
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--tensorboard-dir', type=str, default=Path(os.environ.get('SM_MODULE_DIR')).parent.parent)
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
#     parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--channel', type=int, default=None)
    parser.add_argument("--augment", type=str2bool, nargs='?',
                            const=True, default=False,
                            help="Augment the training data.")
    parser.add_argument("--encoder-freeze", type=str2bool, nargs='?',
                            const=True, default=False,
                            help="Freeze the weights of the encoding layer.")
    parser.add_argument('--backbone', default='resnet152')
    parser.add_argument('--loss', default='bce_jaccard_loss', choices=list(losses.keys()))
    parser.add_argument('--test-prop', type=int, default=10,
                        help="Percentage of images to use for testing")
    parser.add_argument("--downsample", type=str2bool, nargs='?',
                            const=True, default=False,
                            help="Downsample the data to 2km resolution.")    

    args = vars(parser.parse_args())

    main(**args)
