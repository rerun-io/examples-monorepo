from PIL import Image


def add_border_and_mask(
    image: Image.Image,
    zoom_all: int | float = 1.0,
    zoom_left: int | float = 0,
    zoom_right: int | float = 0,
    zoom_up: int | float = 0,
    zoom_down: int | float = 0,
    overlap: int | float = 0,
) -> tuple[Image.Image, Image.Image]:
    """
    Adds a black border around the given image with individual side control and generates a corresponding mask.

    Parameters:
        image (Image.Image): The input PIL image to which the border will be added.
        zoom_all (int | float, optional): Uniform zoom factor applied to all sides. Default is 1.0 (no zoom).
        zoom_left (int | float, optional): Zoom factor for the left border as a fraction of the image width. Default is 0.
        zoom_right (int | float, optional): Zoom factor for the right border as a fraction of the image width. Default is 0.
        zoom_up (int | float, optional): Zoom factor for the top border as a fraction of the image height. Default is 0.
        zoom_down (int | float, optional): Zoom factor for the bottom border as a fraction of the image height. Default is 0.
        overlap (int | float, optional): Overlap between the mask and the original image as a fraction of the image size. Default is 0.

    Returns:
        tuple[Image.Image, Image.Image]:
            - The new image with the black border added.
            - A mask image (mode "L") where the border is white (255) and the original image area is black (0), with overlap applied.
    """
    orig_width, orig_height = image.size

    # Calculate padding for each side (in pixels)
    left_pad = int(orig_width * zoom_left)
    right_pad = int(orig_width * zoom_right)
    top_pad = int(orig_height * zoom_up)
    bottom_pad = int(orig_height * zoom_down)

    # Calculate overlap in pixels
    overlap_left = int(orig_width * overlap)
    overlap_right = int(orig_width * overlap)
    overlap_top = int(orig_height * overlap)
    overlap_bottom = int(orig_height * overlap)

    # If using the all-sides zoom, add it to each side
    if zoom_all > 1.0:
        extra_each_side: float = (zoom_all - 1.0) / 2
        left_pad += int(orig_width * extra_each_side)
        right_pad += int(orig_width * extra_each_side)
        top_pad += int(orig_height * extra_each_side)
        bottom_pad += int(orig_height * extra_each_side)

    # Calculate new dimensions (ensure they're multiples of 32)
    new_width = 32 * round((orig_width + left_pad + right_pad) / 32)
    new_height = 32 * round((orig_height + top_pad + bottom_pad) / 32)

    # Create new image with black border
    bordered_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
    # Paste original image in position
    paste_x = left_pad
    paste_y = top_pad
    bordered_image.paste(image, (paste_x, paste_y))

    # Create mask (white where the border is, black where the original image was)
    mask: Image.Image = Image.new("L", (new_width, new_height), 255)  # White background
    # Paste black rectangle with overlap adjustment
    mask.paste(
        0,
        (
            paste_x + overlap_left,  # Left edge moves right
            paste_y + overlap_top,  # Top edge moves down
            paste_x + orig_width - overlap_right,  # Right edge moves left
            paste_y + orig_height - overlap_bottom,  # Bottom edge moves up
        ),
    )

    return bordered_image, mask


def resize(img: Image.Image, min_mp: float = 0.5, max_mp: float = 2.0) -> Image.Image:
    """
    Resize an image to ensure its megapixel count falls within a specified range
    and its dimensions are multiples of 32.

    Args:
        img: The input PIL Image to be resized.
        min_mp: Minimum allowed megapixels for the output image.
        max_mp: Maximum allowed megapixels for the output image.

    Returns:
        The resized PIL Image with dimensions as multiples of 32
        and megapixels within the specified range.
    """
    width, height = img.size
    current_megapixels: float = (width * height) / 1_000_000

    def _round_to_multiple_of_32(dimension: int) -> int:
        """Round dimension to nearest multiple of 32."""
        return int(32 * round(dimension / 32))

    def _resize_to_dimensions(new_width: int, new_height: int) -> Image.Image:
        """Resize image to new dimensions if they differ from current."""
        if new_width != width or new_height != height:
            return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return img

    # If megapixels are in range, just ensure dimensions are multiples of 32
    if min_mp <= current_megapixels <= max_mp:
        new_width = _round_to_multiple_of_32(width)
        new_height = _round_to_multiple_of_32(height)
        return _resize_to_dimensions(new_width, new_height)

    # Scale image to fit within megapixel range
    target_mp: float = min_mp if current_megapixels < min_mp else max_mp
    scale_factor: float = (target_mp / current_megapixels) ** 0.5

    new_width: int = _round_to_multiple_of_32(int(width * scale_factor))
    new_height: int = _round_to_multiple_of_32(int(height * scale_factor))

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


def process_image(input_image: Image.Image, max_dimension: int = 1920) -> Image.Image:
    """
    Resizes the input PIL Image to fit within a maximum dimension while ensuring both width and height are multiples of 32.

    If the image exceeds the maximum allowed dimension (MAX_DIMENSION), it is scaled down proportionally so that neither width nor height exceeds MAX_DIMENSION, and both dimensions are rounded to the nearest multiple of 32. If the image is already within the allowed size, it is resized (if necessary) to ensure both dimensions are multiples of 32.

    Args:
        input_image (Image.Image): The input PIL Image to be processed.

    Returns:
        Image.Image: The resized PIL Image with dimensions as multiples of 32 and within the maximum allowed size.
    """
    width = input_image.size[0]
    height = input_image.size[1]
    # Calculate scale factor to fit within max_dimension
    scale: float = min(max_dimension / width, max_dimension / height)
    if scale < 1.0:
        new_width = int(32 * round(width * scale / 32))
        new_height = int(32 * round(height * scale / 32))
        input_image = input_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        # Still ensure dimensions are multiples of 32
        input_image = resize(input_image)

    return input_image
