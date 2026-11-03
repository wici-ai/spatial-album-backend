"""Optional bounded FLUX working images; callers retain original output size."""
from PIL import Image
import re

# Conservative tile alignment used in the measured compact deployment.
WORKING_TILE_PIXELS = 64


def parse_max_side(value: str) -> int:
    if not re.fullmatch(r'0|[1-9][0-9]*', value):
        raise ValueError('WICI_INPAINT_MAX_SIDE must be an ASCII integer')
    result = int(value)
    validate_max_side(result)
    return result


def validate_max_side(max_side: int) -> None:
    if max_side != 0 and (max_side < WORKING_TILE_PIXELS or max_side % WORKING_TILE_PIXELS):
        raise ValueError('WICI_INPAINT_MAX_SIDE must be 0 or a positive multiple of 64')


def prepare_fill_input(image, mask, max_side: int):
    validate_max_side(max_side)
    if min(image.size) <= 0 or mask.size != image.size:
        raise ValueError('Fill image and mask must have matching positive dimensions')
    if max_side == 0:
        return image, mask
    scale = min(1.0, max_side / max(image.size))
    size = tuple(max(WORKING_TILE_PIXELS, int(side * scale) // WORKING_TILE_PIXELS * WORKING_TILE_PIXELS)
                 for side in image.size)
    if size == image.size:
        return image, mask
    return (image.resize(size, Image.Resampling.LANCZOS),
            mask.resize(size, Image.Resampling.NEAREST))


def restore_fill_output(output, original_size):
    if output.size == original_size:
        return output
    return output.resize(original_size, Image.Resampling.LANCZOS)
