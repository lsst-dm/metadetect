NO_ATTEMPT=2**0
IMAGE_FLAGS=2**1

NAME_MAP={
    'no_attempt':NO_ATTEMPT,
    NO_ATTEMPT:'no_attempt',
    'image_flags':IMAGE_FLAGS,
    IMAGE_FLAGS:'image_flags',
}

def get_name(val):
    return NAME_MAP[val]