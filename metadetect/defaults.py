DEFAULT_LOGLEVEL = 'INFO'
BMASK_EDGE = 2**30
DEFAULT_IMAGE_VALUES = {
    'image': 0.0,
    'weight': 0.0,
    'seg': 0,
    'bmask': BMASK_EDGE,
    'noise': 0.0,
}

ALLOWED_BOX_SIZES = [
    2,3,4,6,8,12,16,24,32,48,  # noqa
    64,96,128,192,256,  # noqa
    384,512,768,1024,1536,  # noqa
    2048,3072,4096,6144  # noqa
]

# stamp size is not a default, depends on measurement
DEFAULT_MDET_CONFIG = {
    'meas_type': 'wmom',
    'metacal_psf': 'fitgauss',
    'psf_fitter': 'admom',
    'weight_fwhm': 1.2,
    'detect_thresh': 5.0,
    'use_deblended_stamps': False,
    'subtract_sky': False,
}

