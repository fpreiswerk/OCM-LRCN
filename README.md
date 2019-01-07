# OCMDemo.jl
This is demo code for the paper

Preiswerk, Frank, et al. [*"Synthesizing dynamic MRI using long-term recurrent convolutional networks"*](https://arxiv.org/abs/1807.09305), 9th International Conference on Machine Learning in Medical Imaging (MLMI 2018).

## How to use
Python 3 with the following packages is required to run the code:

numpy
h5py
xml
keras
sklearn
matplotlib
cv2

## Download the code

```shell
git clone https://github.com/fpreiswerk/OCM-LRCN
```

## Download the data

To download and extract the data, run the following commands from within the
OCM-MRI directory:

```shell
wget https://www.dropbox.com/s/y0y2o0q8m9z10fp/OCM-LRCN_example_data.tar.gz
tar -xzf OCM-LRCN_example_data.tar.gz
```

## Training a model on the sample datasets

Run the following python script:
```shell
python run_train.py
```

This will train the model, make predictions and save everything to
data/output.

## Rendering the results

Run the script
```shell
python run_results.py
```
to generate images and movies of the results. Everything will also be saved
to the data/output folder.


## Credits
This work was performed with the following co-authors:

- Cheng-Chieh Cheng, Brigham and Women's Hospital, Harvard Medical School, Boston
- Jie Luo, Graduate School of Frontier Sciences, The University of Tokyo, Japan
- Bruno Madore, Brigham and Women's Hospital, Harvard Medical School, Boston
