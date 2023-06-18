import numpy as np
import torch
from scipy.spatial import cKDTree
import torch.nn.functional as F


def image_tensors_to_patches(list_of_image_tensors, patch_size):
    coords = []
    patches = []
    for img_tensor in list_of_image_tensors:
        # If patch size is greater than image size, pad the image with zeros
        image_height , image_width = img_tensor.shape[-2:]
        if patch_size > image_height or patch_size > image_width:
            pad_height = max(patch_size - image_height, 0)
            pad_width = max(patch_size - image_width, 0)
            img_tensor = F.pad(img_tensor, (0, 0, pad_height, pad_width), mode='constant')
        # Compute the number of patches in each dimension
        num_patches_height = int(np.ceil(img_tensor.shape[1] / patch_size))
        num_patches_width = int(np.ceil(img_tensor.shape[2] / patch_size))
        patch_coordinates = []
        for i in range(num_patches_height):
            for j in range(num_patches_width):
                start_h = i * patch_size
                end_h = min(start_h + patch_size, img_tensor.shape[1])
                start_h = end_h - patch_size
                start_w = j * patch_size
                end_w = min(start_w + patch_size, img_tensor.shape[2])
                start_w = end_w - patch_size
                patch = img_tensor[:, start_h:end_h, start_w:end_w]
                patches.append(patch)
                patch_coordinates.append([start_h, start_w, end_h, end_w])
        coords.append(patch_coordinates)
    return torch.stack(patches), coords


def patches_to_image_tensors(patches_tensor, coords):
    list_of_image_tensors = []
    i = 0
    for patch_coordinates in coords:
        patches = patches_tensor[i: i+len(patch_coordinates)]
        max_x = max(coord[2] for coord in patch_coordinates)
        max_y = max(coord[3] for coord in patch_coordinates)
        num_channels = patches.shape[1]
        image = torch.zeros((num_channels, max_x, max_y)).to(patches.device)
        patch_counts = torch.zeros((num_channels, max_x, max_y)).to(patches.device)
        for patch, coordinates in zip(patches, patch_coordinates):
            xs, ys, xe, ye = coordinates
            image[:, xs:xe, ys:ye] += patch
            patch_counts[:, xs:xe, ys:ye] += 1
        list_of_image_tensors.append(image / patch_counts)
        i += len(patch_coordinates)
    return list_of_image_tensors


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


def assign_closest_color_code(segmentation_colors: np.array, np_img: np.array):
    # Convert the segmentation colors to a numpy array
    #segmentation_colors = torch.tensor(segmentation_colors, dtype=torch.uint8)
    tree = cKDTree(segmentation_colors)
    _, indices = tree.query(np_img)
    segmented_image_data = segmentation_colors[indices].reshape(np_img.shape)
    return segmented_image_data


def color_map_to_region_id_map(color_map):
    # Flatten the color map to a 2D array
    flattened_map = color_map.reshape(-1, 3)
    # Get unique colors and their indices
    unique_colors, color_indices = np.unique(flattened_map, axis=0, return_inverse=True)
    # Create a lookup table to map colors to region IDs
    lookup_table = np.arange(len(unique_colors))
    # Reshape the color indices back to the original shape
    region_id_map = lookup_table[color_indices].reshape(color_map.shape[:2])
    return region_id_map


if __name__=='__main__':
    list_of_image_tensors = [torch.randn(3, 256, 256), torch.randn(3, 312, 312), torch.randn(3, 192, 312)]
    patches_tensor, coords = image_tensors_to_patches(list_of_image_tensors, patch_size=256)
    reconstructed_images = patches_to_image_tensors(patches_tensor, coords)
    for recon, img in zip(reconstructed_images, list_of_image_tensors):
        print('diff between reconstructed and original=', recon - img)
    image_size1 = [312, 312]
    patch_size1 = (256, 256)
    image1 = np.random.randint(0, 255, size=(image_size1[0], image_size1[1], 3), dtype=np.uint8)
    patches1, patch_coordinates1 = image_to_patches(image1, patch_size1)
    print("Number of patches:", len(patches1))
    print("Patch coordinates:", patch_coordinates1)
    patches1_tensor = torch.stack([torch.from_numpy(p) for p in patches1]).permute(0, 3, 1, 2)
    patch_coordinates1 = torch.tensor(patch_coordinates1)

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

    color_map = (np.random.rand(10,10)>0.5)*255
    color_map = np.stack([color_map, color_map, color_map])
    color_map = np.transpose(color_map, (1, 2, 0))
    print(color_map)
    region_ids = color_map_to_region_id_map(color_map)
    print(region_ids)
