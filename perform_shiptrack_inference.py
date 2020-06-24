#!/usr/bin/env python
"""
Run inference over a (possibly remote) dataset using a saved tf model
"""
import tensorflow as tf
import numpy as np
# normalise = lambda data: [(d - mean_data)/std_data for d in data]
# normalise = lambda data: (data - 0.5)/0.25
normalise = lambda data: [(d - 0.45)/0.25 for d in data]

# tf_predictor = tf.keras.models.load_model('model/1/')
tf_predictor = tf.saved_model.load('model/1/')
# Check its architecture
#tf_predictor.summary()


def split_array(arr, step_size=440):
    x_len, y_len = arr.shape[0:2]
    x_split_locs = range(step_size, x_len, step_size)
    y_split_locs = range(step_size, y_len, step_size)
    v_splits = np.array_split(arr, x_split_locs, axis=0)
    return sum((np.array_split(v_split, y_split_locs, axis=1) for v_split in v_splits), [])


def process_typed_file(f_key, image_size, path=None, resize=False, channel=None):
    import boto3
    from PIL import ImageOps, Image
    from io import BytesIO


    s3 = boto3.client('s3')
    file_byte_string = s3.get_object(Bucket='imiracli-data', Key=f_key)['Body'].read()

    im_data = Image.open(BytesIO(file_byte_string))
    if (im_data.size[1] not in [2030, 2040]) or (im_data.size[0] != 1354):
        print("Skipping wierd shape: {}".format(im_data.size))
        raise ValueError()
    
    original_shape = im_data.size
    
    if resize:
        im_data = im_data.resize(image_size, resample=Image.BILINEAR)
        padding = 0
    else:  # Pad
        # expand by 160, 80 on each side
        padding = 85 if im_data.size[1] == 2030 else 80
        im_data = ImageOps.expand(im_data, padding)

    # Drop the end of the y axis
    im_data = np.array(im_data)#[:, :1320, :]
    
    if channel is not None:
        im_data = im_data[..., channel]

    return im_data, original_shape


def combine_and_resize_masks(masks, original_size):
    print(masks.shape)
    # TODO: Make the nesting a parameter
    nested_masks = [np.split(m, 5) for m in np.split(masks[..., 0], 8, 0)]
#     print(len(nested_masks))
#     print(nested_masks[0][0].shape)
#     nested_masks = np.reshape(masks, (5, 8))
    
    mask = np.squeeze(np.block(nested_masks))
#     print(mask.shape)
    print(mask.any())
    mask_im = Image.fromarray(mask)

    mask_im = mask_im.resize(original_size, resample=Image.NEAREST)

    # Don't forget to pop off the superfluous color dimension
    new_mask = np.array(mask_im)#[:, :, 0]
    
    return new_mask


def get_ship_track_mask(f):
    data, original_shape = process_typed_file(f, (448*4, 448*3), resize=True)
    norm_data = normalise(data)
    print(original_shape)
    print(data.shape)

    split_data = normalise(split_array(np.concatenate([data[..., 0:1]/255.]*3, axis=-1), 448))
    # Normalise? Anythinf else?
#     print(split_data[0].shape)
    stacked_data = np.stack(split_data)
    print(stacked_data.shape)

    pred = tf_predictor.predict(stacked_data[0:1,...].tolist())
    print(pred)
    prediction = np.array(pred['predictions'])
    
#     for p in prediction:
#         fig, axs = plt.subplots(figsize=(10, 20))
#         axs.imshow(data[..., 0])
#         axs.imshow(p[:,:, 0], alpha=0.5)
#         plt.show()
#     print(prediction.any())
#     print(prediction.shape)
    combined_prediction = combine_and_resize_masks(prediction, original_shape)
    
    combined_data = combine_and_resize_masks(np.stack(split_data), original_shape)
    
#     print(combined_data.shape)
#     print(combined_prediction.any())
#     print(combined_prediction)
    if combined_prediction.any():
        fig, axs = plt.subplots(figsize=(20, 40))
        axs.imshow(combined_data[...], vmin=-2, vmax=2)
        im=axs.imshow(combined_prediction[:,:], alpha=0.5, vmin=0, vmax=1)
#         plt.colorbar(im)
        plt.show()
    return combined_prediction


data_path = 's3://imiracli-data/MODIS_deep_cloud/test_niremi_inference_images/' 
file_path = data_path+"images/MYD021KM.A2006166.1845.061.2018022234106.png"

arr = get_ship_track_mask("MODIS_deep_cloud/test_niremi_inference_images/images/MYD021KM.A2006166.1845.061.2018022234106.png")
