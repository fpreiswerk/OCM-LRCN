import LRCN
import dataset_list

datasets = dataset_list.datasets_LRCN

for dataset in datasets:
    # each dataset contains a path to two acquisitions. In train_predict(), the
    # first acquisition is used for training and the second one for validation.
    # The script also computes the predictions and saves everything.
    LRCN.train.train_predict(dataset)
