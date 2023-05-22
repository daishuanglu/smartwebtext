import numpy as np


def image_to_patches(image, patch_size):
    """
    Crop the image into multiple patches with the given patch size.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        patch_size (tuple): Size of the patches in the format (height, width).

    Returns:
        patches (list): List of patches as NumPy arrays.
        patch_coordinates (list): List of [x, y, w, h] coordinates for each patch.
    """
    image_dims = image.shape
    image_height, image_width = image_dims[:2]
    patch_height, patch_width = patch_size

    # If patch size is greater than image size, pad the image with zeros
    if patch_height > image_height or patch_width > image_width:
        pad_height = max(patch_height - image_height, 0)
        pad_width = max(patch_width - image_width, 0)
        image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
        return [image], [0,0,image_height, image_width]

    # Compute the number of patches in each dimension
    num_patches_height = int(np.ceil(image.shape[0] / patch_height))
    num_patches_width = int(np.ceil(image.shape[1] / patch_width))

    patches = []
    patch_coordinates = []
    for i in range(num_patches_height):
        for j in range(num_patches_width):
            start_h = i * patch_height
            end_h = min(start_h + patch_height, image.shape[0])
            start_h = end_h - patch_height
            start_w = j * patch_width
            end_w = min(start_w + patch_width, image.shape[1])
            start_w = end_w - patch_width
            if len(image_dims) > 2:
                patch = image[start_h:end_h, start_w:end_w, :]
            else:
                patch = image[start_h:end_h, start_w:end_w]
            patches.append(patch)
            patch_coordinates.append([start_h, start_w,  end_h - start_h, end_w - start_w])

    return patches, patch_coordinates


def patches_to_image(patches, patch_coordinates):
    """
    Reconstruct the full image from the list of image patches and their patch coordinates.

    Args:
        patches (list): List of patches as NumPy arrays.
        patch_coordinates (list): List of [x, y, w, h] coordinates for each patch.

    Returns:
        image (numpy.ndarray): Reconstructed full image as a NumPy array.
    """
    max_x = max(coord[0] + coord[2] for coord in patch_coordinates)
    max_y = max(coord[1] + coord[3] for coord in patch_coordinates)
    num_channels = patches[0].shape[2]

    image = np.zeros((max_x, max_y, num_channels))
    patch_counts = np.zeros((max_x, max_y, num_channels))

    for patch, coordinates in zip(patches, patch_coordinates):
        x, y, w, h = coordinates
        image[x:x+w, y:y+h, :] += patch
        patch_counts[x:x+w, y:y+h, :] += 1
    image = np.divide(image, patch_counts, out=np.zeros_like(image), where=patch_counts!=0)
    return image.astype('uint8')


if __name__=='__main__':
    # Example 1
    image_size1 = [312, 312]
    patch_size1 = (256, 256)
    image1 = np.random.randint(0, 255, size=(image_size1[0], image_size1[1], 3), dtype=np.uint8)
    patches1, patch_coordinates1 = image_to_patches(image1, patch_size1)
    print("Number of patches:", len(patches1))
    print("Patch coordinates:", patch_coordinates1)

    # Example 1
    image_size1 = [312, 312]
    patch_size1 = (256, 256)
    image1 = np.random.randint(0, 255, size=(image_size1[0], image_size1[1]), dtype=np.uint8)
    patches1, patch_coordinates1 = image_to_patches(image1, patch_size1)
    print("Number of patches:", len(patches1))
    print("Patch coordinates:", patch_coordinates1)

    # Example 2
    image_size2 = [192, 192]
    patch_size2 = (256, 256)
    image2 = np.random.randint(0, 255, size=(image_size2[0], image_size2[1], 3), dtype=np.uint8)
    patches2, patch_coordinates2 = image_to_patches(image2, patch_size2)
    print("Number of patches:", len(patches2))
    print("Patch coordinates:", patch_coordinates2)

    # Example
    patch_size = [256, 256]

    # Generate random patches and patch coordinates (for demonstration)
    num_patches = 4
    patches = [np.random.randint(0, 255, size=(patch_size[0], patch_size[1], 3), dtype=np.uint8) for _ in
               range(num_patches)]
    patch_coordinates = [[0, 0, patch_size[1], patch_size[0]],
                         [56, 0, patch_size[1], patch_size[0]],
                         [0, 56, patch_size[1], patch_size[0]],
                         [56, 56, patch_size[1], patch_size[0]]]

    # Reconstruct the full image
    reconstructed_image = patches_to_image(patches, patch_coordinates)

    # Display the result
    print("Reconstructed image shape:", reconstructed_image.shape)
