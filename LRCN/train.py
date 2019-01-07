import os
import numpy as np
from time import time
from keras.utils import plot_model

from . import load
from . import processing
from . import models

T_window_learn = 3  # number of seconds to learn from


def train_predict(datasets, predict_all=False):

    print('Training LRCN model')
    print('Training set: {}'.format(datasets[0]))
    print('Test set: {}'.format(datasets[1]))
    outdir = ''.join([os.path.basename(x)+'_' for x in datasets])[0:-1]
    outdir = os.path.join(os.path.dirname(datasets[0]), 'output', outdir)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    train, test = load.load_train_and_test_data(datasets)
    N_TR_learn = int(round(T_window_learn /
                     train['t_ocm'][1]-train['t_ocm'][0]))
    n_planes = 1

    # one minute of training, 30 seconds of testing
    t_plane_s = (train['t_mr'][0][1] - train['t_mr'][0][0]) / n_planes
    n_train = int(60 / t_plane_s)
    n_test = int(30 / t_plane_s)

    n_train_valid = [0] * 2
    n_test_valid = [0] * 2

    # chop data to right length
    for plane in range(n_planes):

        valid_mr_train_inds = np.where(train['mr2us'][plane] >= N_TR_learn)[0]
        n_train_valid[plane] = np.minimum(n_train, len(valid_mr_train_inds))
        valid_mr_train_inds = valid_mr_train_inds[0:n_train_valid[plane]]
        train['mr2us'][plane] = train['mr2us'][plane][valid_mr_train_inds]
        train['t_mr'][plane] = train['t_mr'][plane][valid_mr_train_inds]
        train['mri'][plane] = train['mri'][plane][valid_mr_train_inds, :, :]

        valid_mr_test_inds = np.where(test['mr2us'][plane] >= N_TR_learn)[0]
        n_test_valid[plane] = np.minimum(n_test, len(valid_mr_test_inds))
        valid_mr_test_inds = valid_mr_test_inds[0:n_test_valid[plane]]
        test['mr2us'][plane] = test['mr2us'][plane][valid_mr_test_inds]
        test['t_mr'][plane] = test['t_mr'][plane][valid_mr_test_inds]
        test['mri'][plane] = test['mri'][plane][valid_mr_test_inds, :, :]

    train_pcas = list()
    train_mri_pca = list()
    test_mri_pca = list()
    train_pca_scale_factors = list()

    # Apply PCA transform to training and test MR images
    for plane in range(n_planes):
        # training images
        train['mri'][plane] = processing.hist_match_mri(
                                train['mri'][plane],
                                train['mri'][plane][0, :, :])
        imgs_this_plane_flat = processing.flatten_mri(train['mri'][plane])
        pca = processing.compute_pca(imgs_this_plane_flat, 10)
        pca_tmp = processing.pca_transform(imgs_this_plane_flat, pca)
        pca_scale_factor = np.std(pca_tmp[:, 0])
        train_mri_pca.append(pca_tmp / pca_scale_factor)
        train_pcas.append(pca)
        train_pca_scale_factors.append(pca_scale_factor)

        # test images
        test['mri'][plane] = processing.hist_match_mri(
                                        test['mri'][plane],
                                        train['mri'][plane][0, :, :])
        imgs_this_plane_flat = processing.flatten_mri(test['mri'][plane])
        pca_tmp = processing.pca_transform(imgs_this_plane_flat, pca)
        test_mri_pca.append(pca_tmp / pca_scale_factor)

    # format training and test data into a suitable format for training
    X_train_all_planes = list()
    Y_train_all_planes = list()
    X_test_all_planes = list()
    Y_test_all_planes = list()
    for plane in range(n_planes):
        # format training data
        X_train = np.ndarray((n_train_valid[plane], N_TR_learn,
                             train['ocm'].shape[1]))
        for i in range(n_train_valid[plane]):
            idx_ocm = np.argmin(np.abs(train['t_ocm']-train['t_mr'][plane][i]))
            X_train[i, :, :] = np.squeeze(
                            train['ocm'][idx_ocm-N_TR_learn+1:idx_ocm+1, :])
        X_train_all_planes.append(X_train)
        Y_train_all_planes.append(train_mri_pca[plane])

        # format test data
        X_test = np.ndarray((n_test_valid[plane], N_TR_learn,
                             test['ocm'].shape[1]))
        for i in range(n_test_valid[plane]):
            idx_ocm = np.argmin(np.abs(test['t_ocm']-test['t_mr'][plane][i]))
            X_test[i, :, :] = np.squeeze(
                            test['ocm'][idx_ocm-N_TR_learn+1:idx_ocm+1, :])
        X_test_all_planes.append(X_test)
        Y_test_all_planes.append(test_mri_pca[plane])

    # train the model
    n_epochs = 1000
    input_shape = X_train_all_planes[0].shape[1:]
    output_dim = Y_train_all_planes[0].shape[1]
    all_models = list()
    for plane in range(n_planes):
        model = models.LRCN(input_shape, output_dim)
        start = time()
        model.fit(X_train_all_planes[plane], Y_train_all_planes[plane],
                  epochs=n_epochs, batch_size=n_train_valid[plane],
                  shuffle=False,  verbose=2)
        end = time()
        print("Training time was {:.2f} s.".format(end - start))

        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        model.save(os.path.join(
                    outdir, 'model_LRCN_plane_' + str(plane) + '.h5'))
        plot_model(model, to_file=os.path.join(outdir, 'model.png'))
        all_models.append(model)

    # predict test images
    all_predictions = list()
    all_gt = list()
    for plane in range(n_planes):
        predictions = all_models[plane].predict(
                                X_test_all_planes[plane],
                                batch_size=X_test_all_planes[plane].shape[0])
        predictions = predictions * train_pca_scale_factors[plane]
        gt = Y_test_all_planes[plane] * train_pca_scale_factors[plane]
        all_predictions.append(predictions)
        all_gt.append(gt)

    # predict at high temporal rate - one image for each OCM signal (100 Hz)
    all_predictions_hs = list()
    for plane in range(n_planes):
        predictions = np.zeros((test['ocm'].shape[0],
                                Y_train_all_planes[plane].shape[1]))
        # formatting all OCM data for network input at once might be a bit too
        # memory intensive - split it up instead
        n_batches_predict = 10
        batch_len = int((test['ocm'].shape[0]-N_TR_learn)/n_batches_predict)
        for batch in range(n_batches_predict):
            ocm_test_inds = np.arange(batch*batch_len,
                                      batch*batch_len+batch_len) + N_TR_learn
            X_test = np.ndarray((len(ocm_test_inds), N_TR_learn,
                                test['ocm'].shape[1]))
            for idx, val in enumerate(ocm_test_inds):
                X_test[idx, :, :] = np.squeeze(
                    test['ocm'][int(val)-N_TR_learn:int(val), :])
            batch_prediction = all_models[plane].predict(
                                            X_test,
                                            batch_size=n_train_valid[plane])
            predictions[batch*batch_len+N_TR_learn:(batch+1)
                        * batch_len+N_TR_learn, :] = batch_prediction

        all_predictions_hs.append(predictions * train_pca_scale_factors[plane])

    # timing analysis - run one batch three times, with PCA recon
    start = time()
    foo = all_models[0].predict(X_test, batch_size=n_train_valid[plane])
    train_pcas[0].inverse_transform(foo * train_pca_scale_factors[0])
    foo = all_models[0].predict(X_test, batch_size=n_train_valid[plane])
    train_pcas[0].inverse_transform(foo * train_pca_scale_factors[0])
    foo = all_models[0].predict(X_test, batch_size=n_train_valid[plane])
    train_pcas[0].inverse_transform(foo * train_pca_scale_factors[0])
    end = time()
    print("Recon time was {:.2f} s for {} frames, i.e., {:.3f} ms per frame".
          format(end - start, X_test.shape[0]*3,
                 (end - start)/float(X_test.shape[0]*3)*1000))

    # save everything
    np.save(os.path.join(outdir, 'highspeed_prediction.npy'),
            all_predictions_hs)
    np.save(os.path.join(outdir, 'all_prediction.npy'), all_predictions)
    np.save(os.path.join(outdir, 'all_gt.npy'), all_gt)
    np.save(os.path.join(outdir, 'pcas.npy'), train_pcas)
    print('Model and results saved in {}'.format(outdir))
