import os
import numpy as np
import scipy.misc
from scipy.misc import imsave

from LRCN import load, videoRender
import dataset_list

datasets = dataset_list.datasets_LRCN

for num, dataset in enumerate(datasets):

    indir = ''.join([os.path.basename(x)+'_' for x in dataset])[0:-1]
    indir = os.path.join(os.path.dirname(dataset[0]), 'output', indir)

    ds_name = os.path.basename(os.path.dirname(dataset[0]))
    all_predictions = np.load(os.path.join(indir, 'all_prediction.npy'))
    all_gt = np.load(os.path.join(indir, 'all_gt.npy'))
    pcas = np.load(os.path.join(indir, 'pcas.npy'), encoding='latin1')

    n_planes = 1

    # load original data to render the uncompressed (pca) test images as movie
    train, test = load.load_train_and_test_data(dataset)

    # render the low-speed predictions (one for each test MR image)
    for plane in range(n_planes):

        pca = pcas[plane]
        recon_test = pca.inverse_transform(all_predictions[plane])
        recon_test = np.reshape(recon_test, (recon_test.shape[0], 192, 192))
        test_gt = pca.inverse_transform(all_gt[plane])
        test_gt = np.reshape(test_gt, (test_gt.shape[0], 192, 192))

        recon_test[recon_test < 0] = 0
        test_gt[test_gt < 0] = 0
        test_gt[test_gt == np.inf] = 0
        ma = np.max(test_gt)  # normalize with the maximum of the ground-truth
        recon_test = recon_test / (ma)
        test_gt = test_gt / (ma)
        test_gt *= 0.75  # matching KDE histogram profile as good as it gets
        recon_test *= 0.75  # matching KDE histogram profile as good as it gets

        # '3 - plane' equals the number of invalid MR frames at the beginning
        # (3 for plane 1, 2 for plane 2)
        test_gt = test['mri'][0][3-plane:3-plane+25, :, :]
        test_gt[test_gt < 0] = 0
        test_gt = test_gt / (ma)  # Normalize ground-truth
        diff = (recon_test - test_gt) / 2 + 0.5

        comparison = np.concatenate((recon_test, test_gt, diff), axis=1)
        comparison = np.concatenate((comparison, comparison, comparison),
                                    axis=2)
        text = "LRCN output       Acquired            Difference"
        vid_fn = os.path.join(indir,
                              ds_name + '_comparison_plane{}_'.format(plane))
        videoRender.render_movie(comparison, vid_fn + '.mp4', 0.6, text)
        os.system('ffmpeg -y -i ' + vid_fn + '.mp4'
                  + ' -filter:v \"crop=576:192:0:0\" ' + vid_fn[0:-1] + '.mp4')
        print('Native speed comparison video saved to {}'.format(vid_fn))

        idx_mid = int(test_gt.shape[1] / 2)

        mmode_test_gt = test_gt[:, idx_mid, :]
        mmode_recon_test = recon_test[:, idx_mid, :]
        mmode_diff = mmode_recon_test - mmode_test_gt

        divider = np.transpose(np.ndarray((2, mmode_test_gt.shape[1])))
        divider.fill(np.max(mmode_test_gt))
        mmode_comparison = np.concatenate((np.transpose(mmode_recon_test),
                                          divider, np.transpose(mmode_test_gt),
                                          divider, np.abs(np.transpose(
                                                        mmode_diff))), axis=1)

        imsave(os.path.join(indir,
               'mmode_test_gt_plane'+str(plane)+'.png'), mmode_test_gt)
        imsave(os.path.join(indir,
               'mmode_recon_test_plane'+str(plane)+'.png'), mmode_recon_test)
        imsave(os.path.join(indir,
               'mmode_diff_plane'+str(plane)+'.png'), mmode_diff)
        imsave(os.path.join(indir,
               'mmode_comparison_plane'+str(plane)+'.png'), mmode_comparison)

        # pick a specific image, save reconstruction and ground-truth as image
        example_image_gt = np.transpose(np.squeeze(test_gt[15, :, :]))
        example_image_rec = np.transpose(np.squeeze(
                                         recon_test[15, :, :]))
        imsave(os.path.join(indir,
               'example_image_gt_plane'+str(plane)+'.png'), example_image_gt)
        imsave(os.path.join(indir,
               'example_image_rec_plane'+str(plane)+'.png'), example_image_rec)
        imsave(os.path.join(indir,
               'example_image_diff_plane'+str(plane)+'.png'),
               np.abs(example_image_rec-example_image_gt))

        print('M-mode images and example images saved to {}'.format(indir))

    # render high-speed predictions (at the speed of OCM)
    highspeed_prediction = np.load(os.path.join(indir,
                                   'highspeed_prediction.npy'))
    for plane in range(n_planes):
        # make high speed prediction movie
        pca = pcas[plane]

        # Reconstruct 5000 images using inverse PCA
        highspeed_prediction_ = pca.inverse_transform(
                        highspeed_prediction[plane][0:5000, :])
        highspeed_prediction_ = np.reshape(highspeed_prediction_,
                                           (highspeed_prediction_.shape[0],
                                            192, 192))
        m = comparison.mean()
        std = comparison.std()
        highspeed_prediction_
        highspeed_prediction_[highspeed_prediction_ < m - 2*std] = np.inf
        highspeed_prediction_ = highspeed_prediction_ \
            - np.min(highspeed_prediction_)
        highspeed_prediction_[highspeed_prediction_ == np.inf] = 0
        highspeed_prediction_ = highspeed_prediction_ \
            / np.max(highspeed_prediction_)
        vid_fn = os.path.join(
            indir, ds_name + '_highspeed_prediction_plane{}.mp4'.format(plane))
        videoRender.render_movie(
            highspeed_prediction_, vid_fn, 100,  "Highspeed synthetic MRI")
        print('Highspeed synthetic MRI video saved to {}'.format(vid_fn))

        # This mmode pos is chosen to match the KDE dataset in MRM paper
        mmode_pos = highspeed_prediction_.shape[1] - (103-1)
        fn_mmode = os.path.join(indir,
                                'mmode_highspeed_plane{}.png'.format(plane))
        scipy.misc.imsave(
            os.path.join(indir, 'mmode_highspeed_plane{}.png'.format(plane)),
            np.squeeze(highspeed_prediction_[:, mmode_pos, :]))
        print('Highspeed synthetic MRI M-mode image saved to {}'.
              format(fn_mmode))
