import numpy as np


def norm01(x):
    return (x - x.min())/(x.max() - x.min())


def pw_cos_sim(x, y):
    x_normalized = x / np.linalg.norm(x, axis=1, keepdims=True)
    y_normalized = y / np.linalg.norm(y, axis=1, keepdims=True)

    # Compute pairwise cosine similarity
    similarity_matrix = np.dot(x_normalized, y_normalized.T)
    return similarity_matrix


def vec2mat_cos_sim(vec, mat):
    vec=np.atleast_2d(vec)
    mat=mat.transpose()
    p1 = vec.dot(mat)
    mat_norm=np.sqrt(np.einsum('ij,ij->j',mat,mat))
    mat_norm[mat_norm==0]=1e-3
    vec_norm=np.linalg.norm(vec)
    vec_norm=1e-3 if vec_norm==0 else vec_norm
    out1 = p1 / (mat_norm*vec_norm)
    #print(np.abs(np.linalg.norm(mat,axis=0)-mat_norm).max())
    return out1

# Function to compute the Intersection over Union (IoU) between two regions
def calculate_iou(region_a, region_b):
    intersection = np.logical_and(region_a, region_b)
    union = np.logical_or(region_a, region_b)
    iou = np.sum(intersection) / np.sum(union)
    return iou


if __name__ == '__main__':
    # Example input arrays
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1,1,1]])  # Shape: (3, 3)
    y = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1], [1,1,1]])  # Shape: (3, 3)
    sim = pw_cos_sim(x, y)
    print(sim)

    for vec in x:
        print(vec2mat_cos_sim( vec, y))

