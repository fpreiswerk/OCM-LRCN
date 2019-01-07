import numpy as np
from sklearn.decomposition import PCA


def compute_pca(mri, n_pca):
    pca = PCA(n_components=n_pca)
    pca.fit(np.transpose(mri))
    return pca


def pca_transform(mri, pca):

    images_pca = np.ndarray((mri.shape[1], pca.n_components))
    images_pca = pca.transform(np.transpose(mri))
    return images_pca


# def hist_match_mri(mri):
#     for i in range(1, mri.shape[0]):
#         mri[i, :, :] = hist_match(mri[i, :, :], mri[0, :, :])
#     return mri


def hist_match_mri(mri, reference):
    for i in range(1, mri.shape[0]):
        mri[i, :, :] = hist_match(mri[i, :, :], reference)
    return mri


def flatten_mri(mri):
    images_flat = np.transpose(
        np.reshape(mri[:, :, :], (mri[0:].shape[0], -1)))
    return images_flat


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)
