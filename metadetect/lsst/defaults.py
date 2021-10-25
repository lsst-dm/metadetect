from copy import deepcopy


DEFAULT_WEIGHT_FWHMS = {
    'wmom': 1.2,
    'ksigma': 2.0,
    'pgauss': 2.0,
    'em': -1,  # not using moments currently
}

DEFAULT_STAMP_SIZES = {
    'wmom': 32,
    # TODO determine a good value for this. We used 48 in DES
    # which would be 64 for lsst
    'ksigma': 64,
    'am': 64,
    'pgauss': 64,  # TODO would smaller be OK since does not ring?
    'em': -1,  # no stamps used currently
}

# threshold for detection
DEFAULT_THRESH = 5.0
# refind the center in lsst.measure.measure
DEFAULT_FIND_CEN = False

# deblending settings
DEFAULT_DEBLEND = False
DEFAULT_DEBLENDER = 'scarlet'

# config for shredder deblender
DEFAULT_SHREDDER_CONFIG = {
    'psf_ngauss': 3,
    'init_model': 'exp',
    'miniter': 40,
    'maxiter': 500,
    'flux_miniter': 20,
    'flux_maxiter': 500,
    'tol': 0.001,
}

# whether to find and subtract the sky, happens before metacal
DEFAULT_SUBTRACT_SKY = False

# config for fitting the original psfs
DEFAULT_PSF_CONFIG = {
    'model': 'am',
    'ntry': 4,
}

# Control of the metacal process
DEFAULT_METACAL_CONFIG = {
    "use_noise_image": True,
    "psf": "fitgauss",
}

# detection config, this may expand
DEFAULT_DETECT_CONFIG = {
    'thresh': DEFAULT_THRESH,
}

# the weight subconfig and the stamp_size defaults we be filled in
# programatically based on the measurement_type
DEFAULT_MDET_CONFIG = {
    'meas_type': 'wmom',
    'subtract_sky': DEFAULT_SUBTRACT_SKY,
    'detect': deepcopy(DEFAULT_DETECT_CONFIG),
    'deblend': DEFAULT_DEBLEND,
    'deblender': DEFAULT_DEBLENDER,
    'shredder_config': None,
    'psf': deepcopy(DEFAULT_PSF_CONFIG),
    'metacal': deepcopy(DEFAULT_METACAL_CONFIG),
    'find_cen': DEFAULT_FIND_CEN,
    'weight': None,
    'stamp_size': None,
}
