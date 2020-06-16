import argparse, os
import numpy as np

import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import *
from segmentation_models.metrics import iou_score

losses = {'jaccard_loss': jaccard_loss,
          'dice_loss': dice_loss,
          'binary_focal_loss': binary_focal_loss,
          'binary_crossentropy': binary_crossentropy,
          'bce_dice_loss': bce_dice_loss,
          'bce_jaccard_loss': bce_jaccard_loss,
          'binary_focal_dice_loss': binary_focal_dice_loss,
          'binary_focal_jaccard_loss': binary_focal_jaccard_loss}

def random_pad(vec, pad_width, *_, **__):
    print(vec.shape)
    vec[:pad_width[0]] = np.random.uniform(size=pad_width[0])
    vec[vec.size - pad_width[1]:] = np.random.uniform(size=pad_width[1])


def get_data_flow(data, labels, subset, batch_size=1):
    # this is the augmentation configuration we will use for training
    from keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)

    generator = datagen.flow(
        data, y=labels,
        batch_size=batch_size if subset == 'training' else 1,
        subset=subset)

    return generator


def fit_model(model, training_data, labels, batch_size=1, n_gpus=1, epochs=1, augment=True, tensorboard_dir=''):

    tensorboard = TensorBoard(log_dir=tensorboard_dir, histogram_freq=0,
                              write_images=True, write_graph=False)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=5e-7, verbose=1)

    if augment:
        train_generator = get_data_flow(training_data, labels, batch_size=batch_size * n_gpus, subset='training')
        validation_generator = get_data_flow(training_data, labels, batch_size=batch_size * n_gpus, subset='validation')

        history = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=len(train_generator),
                                      epochs=epochs,
                                      validation_steps=1,
                                      validation_data=validation_generator,
                                      callbacks=[tensorboard, reduce_lr],
                                      # use_multiprocessing=True, workers=8,
                                      verbose=1)
    else:
        history = model.fit(x=training_data, y=labels, validation_split=0.181818, verbose=1,
                            batch_size=batch_size * n_gpus, epochs=epochs,
                            callbacks=[tensorboard, reduce_lr])

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
        
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--tensorboard-dir', type=str, default=os.environ.get('SM_MODULE_DIR'))
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

    args, _ = parser.parse_known_args()

    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size
    gpu_count = args.gpu_count
    model_dir = args.model_dir
    training_dir = args.training
#     validation_dir = args.validation
    channel = args.channel

    all_data = np.load(os.path.join(training_dir, 'data.npz'))['arr_0']
    all_labels = np.load(os.path.join(training_dir, 'labels.npz'))['arr_0']

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
#     data_norm = (padded_data - 0.45) / 0.25
    data_norm = padded_data

    n_test = (data_norm.shape[0] // 100) * args.test_prop
    x_test, y_test = data_norm[:n_test, ...], padded_labels[:n_test, ...]
    x_train, y_train = data_norm[n_test:, ...], padded_labels[n_test:, ...]

    print(x_train.shape[0], 'train/val samples')
    print(x_test.shape[0], 'test samples')

    # input image dimensions
    img_rows, img_cols = 448, 448

    # Tensorflow needs image channels last, e.g. (batch size, width, height, channels)
    K.set_image_data_format('channels_last')
    print(K.image_data_format())

    # preprocess input
    preprocess_input = get_preprocessing(args.backbone)
    x_train = preprocess_input(x_train)

    # define model
    model = Unet(args.backbone, encoder_weights='imagenet', encoder_freeze=args.encoder_freeze, classes=1, activation='sigmoid')
    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)

    print(model.summary())

    model.compile(Adam(lr=lr), loss=losses[args.loss], metrics=[iou_score])

    history = fit_model(model, x_train, y_train, batch_size=batch_size, n_gpus=gpu_count,
                        epochs=epochs, augment=args.augment)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss    :', score[0])
    print('Test accuracy:', score[1])

    # save Keras model for Tensorflow Serving
    sess = K.get_session()
    tf.saved_model.simple_save(
        sess,
        os.path.join(model_dir, 'model/1'),
        inputs={'inputs': model.input},
        outputs={t.name: t for t in model.outputs})
