import cv2
import numpy as np
import os
import openslide
import skimage
import skimage.morphology


def remove_small_hole(mask, h_size=10):
    """remove the small hole

    Args:
        mask (_type_): a binary mask, can be 0-1 or 0-255
        h_size (int, optional): min_size of the hole

    Returns:
        mask
    """
    value = np.unique(mask)
    if len(value) > 2:
        return None
    pre_mask_rever = mask <= 0
    pre_mask_rever = skimage.morphology.remove_small_objects(
        pre_mask_rever, min_size=h_size
    )
    mask[pre_mask_rever <= 0] = np.max(mask)
    return mask


def preprocess_ndpi(
    ndpi_path, output_folder, min_size=100, slice_size=224, tissue_threshold=0.3
):
    # Open NDPI file
    slide = openslide.OpenSlide(ndpi_path)

    # Get level 2 dimensions
    level_dimensions = slide.level_dimensions[3]

    # Read level 2 image
    level2_image = slide.read_region((0, 0), 2, level_dimensions)
    level2_image = cv2.cvtColor(np.array(level2_image), cv2.COLOR_RGBA2RGB)

    # Convert to grayscale
    gray = cv2.cvtColor(level2_image, cv2.COLOR_BGR2GRAY)

    # Remove noise using a Gaussian filter
    gray = cv2.GaussianBlur(gray, (35, 35), 0)

    # Otsu thresholding and mask generation
    ret, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove small tissue regions
    thresh_otsu = remove_small_hole(thresh_otsu, min_size * 4)
    thresh_otsu = 255 - thresh_otsu  # Invert mask
    # Remove small blank regions
    thresh_otsu = remove_small_hole(thresh_otsu, min_size)

    # Find contours
    contours, _ = cv2.findContours(
        thresh_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each contour
    for i, contour in enumerate(contours):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Center crop to 224x224
        cx = x + w // 2
        cy = y + h // 2
        x = cx - slice_size // 2
        y = cy - slice_size // 2

        # Ensure slice is within image boundaries
        x = max(0, min(level_dimensions[0] - slice_size, x))
        y = max(0, min(level_dimensions[1] - slice_size, y))

        # Extract region from level 2 image
        region = level2_image[y : y + slice_size, x : x + slice_size]

        # Check tissue area proportion
        region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, region_thresh = cv2.threshold(
            region_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        tissue_area = np.sum(region_thresh == 0)
        tissue_proportion = tissue_area / (slice_size * slice_size)

        if tissue_proportion < tissue_threshold:
            continue  # Skip this slice if tissue proportion is too low

        # Save region as an image
        output_path = os.path.join(output_folder, f"slice_{i}.jpg")
        cv2.imwrite(output_path, region)

    # Close NDPI file
    slide.close()


# Example usage
ndpi_path = "./PKG - UPENN-GBM_v2/NDPI_images/7316UP-109.ndpi"  # Path to your NDPI file
output_folder = "output_slices"  # Folder to save the slices
min_size = 100  # Minimum size of the blank region to be removed
slice_size = 224  # Size of each slice
preprocess_ndpi(ndpi_path, output_folder, min_size, slice_size)
