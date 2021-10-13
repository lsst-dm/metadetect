NO_ATTEMPT = 2**0
IMAGE_FLAGS = 2**1
PSF_FAILURE = 2**2
OBJ_FAILURE = 2**3
NOMOMENTS_FAILURE = 2**4
BAD_BBOX = 2**5
BBOX_HITS_EDGE = 2**6
ZERO_WEIGHTS = 2**7
CENTROID_FAIL = 2**8
DEBLEND_FAIL = 2**9

NAME_MAP = {
    # no attempt was made to measure this object, usually
    # due to a previous step in the code fails.  E.g. this
    # will be set for the psf flags if there are IMAGE_FLAGS
    # for the image

    'no_attempt': NO_ATTEMPT,
    NO_ATTEMPT: 'no_attempt',

    # there was an issue with the image data
    'image_flags': IMAGE_FLAGS,
    IMAGE_FLAGS: 'image_flags',

    # psf fitting failed
    PSF_FAILURE: 'psf_failure',
    'psf_failure': PSF_FAILURE,

    # object fitting failed
    OBJ_FAILURE: 'obj_failure',
    'obj_failure': OBJ_FAILURE,

    # moment measurement failed
    NOMOMENTS_FAILURE: 'nomoments_failure',
    'nomoments_failure': NOMOMENTS_FAILURE,

    # there was a problem with the bounding box
    BAD_BBOX: 'bad_bbox',
    'bad_bbox': BAD_BBOX,

    BBOX_HITS_EDGE: 'bbox_hits_edge',
    'bbox_hits_edge': BBOX_HITS_EDGE,

    ZERO_WEIGHTS: 'zero_weights',
    'zero_weights': ZERO_WEIGHTS,

    CENTROID_FAIL: 'centroid_fail',
    'centroid_fail': CENTROID_FAIL,
}


def get_name(val):
    return NAME_MAP[val]
