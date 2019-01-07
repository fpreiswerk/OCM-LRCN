import os
import h5py
import numpy as np
import xml.etree.ElementTree
from sklearn.preprocessing import StandardScaler


def load_train_and_test_data(datasets):

    mr2us = list()
    t_mr = list()
    for idx, dataset in enumerate(datasets):
        print(dataset)
        ocm_, t_ocm, mri_, mr2us_, TR = load_dataset(dataset)
        ocm_scaler = StandardScaler()
        dim_old = ocm_.shape[1]
        ocm = ocm_scaler.fit_transform(np.reshape(ocm_, (-1, 1)))
        ocm = np.reshape(ocm, (-1, dim_old))
        t_mr_ = mr2us_ * TR / 1000.0
        t_mr = list()
        mr2us = list()
        mri = list()
        n_planes = mri_.shape[1]
        for plane in range(n_planes):
            t_mr.append(t_mr_[plane::n_planes])
            mr2us.append(mr2us_[plane::n_planes])
            mri.append(np.squeeze(mri_[:, plane, :, :]))
        if idx == 0:
            train = {'ocm': ocm, 't_ocm': t_ocm, 'mri': mri, 't_mr': t_mr,
                     'mr2us': mr2us}
        else:
            test = {'ocm': ocm, 't_ocm': t_ocm, 'mri': mri, 't_mr': t_mr,
                    'mr2us': mr2us}

    return train, test


def load_dataset(dataset):
    ds_root = dataset[0:dataset.rfind('/')]

    # load OCM properties
    fn = os.path.join(ds_root, 'config.xml')
    e = xml.etree.ElementTree.parse(fn).getroot()
    TR = int(e.findall('properties')[0][0].get('value'))

    # load OCM
    fn = os.path.join(dataset, 'ocm_phase.h5')
    dataset_name = '/vz_'
    time_name = 't'
    file = h5py.File(fn, 'r')
    ocm = file[dataset_name][:]
    t_ocm = np.squeeze(file[time_name][:])*100 * TR/1000.0
    print(ocm.shape)
    file.close()

    # load MRI
    fn = os.path.join(dataset, 'mr_data.h5')
    dataset_name = '/mr_data/I'
    file = h5py.File(fn, 'r')
    data = file[dataset_name]
    mri = data[:]

    mri *= 1e3  # for numerical stability later on
    file.close()

    # load us2mr
    fn = os.path.join(dataset, 'mr2us.h5')
    dataset_name_1 = '/mr2us/plane1'
    dataset_name_2 = '/mr2us/plane2'
    file = h5py.File(fn, 'r')
    data1 = file[dataset_name_1]
    data2 = file[dataset_name_2]
    mr2us1 = np.squeeze(data1[:])
    mr2us2 = np.squeeze(data2[:])
    file.close()
    mr2us = np.array(list(zip(mr2us1, mr2us2)))
    mr2us = np.reshape(mr2us, -1)

    return ocm, t_ocm, mri, mr2us, TR
