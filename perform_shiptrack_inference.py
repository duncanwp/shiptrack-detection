#!/usr/bin/env python
"""
Run inference over a (possibly remote) dataset using a saved tf model
"""
import matplotlib
matplotlib.use('agg')

import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt

# normalise = lambda data: [(d - mean_data)/std_data for d in data]
# normalise = lambda data: (data - 0.5)/0.25
normalise = lambda data: [(d - 0.45)/0.25 for d in data]


def split_array(arr, step_size=440):
    x_len, y_len = arr.shape[0:2]
    x_split_locs = range(step_size, x_len, step_size)
    y_split_locs = range(step_size, y_len, step_size)
    v_splits = np.array_split(arr, x_split_locs, axis=0)
    return sum((np.array_split(v_split, y_split_locs, axis=1) for v_split in v_splits), [])


def process_typed_file(file, image_size, resize=False, channel=None):
    from PIL import ImageOps, Image

    im_data = Image.open(file)
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

    # Drop the alpha channel
    im_data = np.array(im_data)[:, :, 0:3]
    
    if channel is not None:
        im_data = im_data[..., channel]

    return im_data, original_shape


def resize(arr, target_size, bilinear=False):
    from PIL import Image
    
    resampler = Image.BILINEAR if bilinear else Image.NEAREST
    
    arr_im = Image.fromarray(arr)

    arr_im = arr_im.resize(target_size, resample=resampler)
    
    new_arr = np.array(arr_im)
    return new_arr

def combine_and_resize_masks(masks, original_size):
    from PIL import Image

    #print(masks.shape)
    # TODO: Make the nesting a parameter
    nested_masks = [np.split(m, 4) for m in np.split(masks[..., 0], 3, 0)]
#     print(len(nested_masks))
#     print(nested_masks[0][0].shape)
#     nested_masks = np.reshape(masks, (5, 8))
    #print([m.shape for m in nested_masks])
    
    mask = np.squeeze(np.block(nested_masks))
    #print(mask.shape)
    #print(mask.any())

    new_mask = resize(mask, original_size)
    
    return new_mask


def get_ship_track_mask(f):
    data, original_size = process_typed_file(f, (448*4, 448*3), resize=True)
    #plt.imshow(data)
    #plt.show()
    # norm_data = normalise(data)
    print("Original size: " + str(original_size))
    print("Re-shaped shape: " + str(data.shape))

    # split_data = normalise(split_array(np.concatenate([data[..., 0:1]/255.]*3, axis=-1), 448))
    split_data = split_array(data, 448)

    prediction = tf_predictor(inputs=tf.constant(np.stack(split_data), dtype='float32'))

    prediction = prediction['sigmoid_1/concat:0'] > 0.5

    combined_prediction = combine_and_resize_masks(prediction, original_size)

    #plt.imshow(combined_prediction)
    #plt.show()
    
    # Test the recombination is working correctly
    #combined_data = combine_and_resize_masks(np.stack(split_data), original_size)
    #print("Testing data combination:")
    #print("Recombined shape: " + str(combined_data.shape))

    #plt.imshow(combined_data)
    #plt.show()
    #print("Reshaped data: " + str(data[100,100,0]))
    #print("Recomnbined data: " + str(combined_data[100,100]))

    original_size_data = resize(data, original_size, bilinear=True)

    return original_size_data, combined_prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model', help="The model directory to use")
    parser.add_argument('infiles', nargs='*')
    parser.add_argument('--infile', help="Input file", type=argparse.FileType('r'))
    parser.add_argument('--show', action='store_true')
    parser.add_argument('-o', '--outfile', help="Output name", default='mask')

    args = parser.parse_args()

    # tf_predictor = tf.keras.models.load_model(args.model+'/1/')
    model = tf.saved_model.load(args.model+'/1/')
    tf_predictor = model.signatures["serving_default"]
    print(tf_predictor)
    # Check its architecture
    # tf_predictor.summary()

    if args.infile is not None:
        print("Reading filelist from {}".format(args.infile))
        infiles = args.infile.readlines()
    else:
        infiles = args.infiles

    for f in infiles:
        data, mask = get_ship_track_mask(f)

        print(data.shape)
        print(mask.any())
        print(mask.shape)
        if mask.any():
            if args.show:
                fig, axs = plt.subplots(figsize=(20, 40))
                axs.imshow(data, vmin=0., vmax=1.)
                im=axs.imshow(mask, alpha=0.5, vmin=0, vmax=1)
        #         plt.colorbar(im)
                plt.savefig(f"{f[:-4]}_{args.outfile}.png")
            np.savez_compressed(f"{f[:-4]}_{args.outfile}.npz", mask=mask)
        else:
            print(f"No shiptracks found in {f}")
