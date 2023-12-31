# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi (federico@disneyresearch.com)
# Adapted from DAVIS 2016 (Federico Perazzi)
# ----------------------------------------------------------------------------

import numpy as np
from skimage.morphology import binary_dilation, disk
import warnings


def mean(X):
  """
  Compute average ignoring NaN values.
  """
  return np.nanmean(X)


def recall(X, threshold=0.5):
  """
  Fraction of values of X scoring higher than 'threshold'
  """
  return mean(np.array(X)>threshold)


def decay(X, n_bins=4):
  """
  Performance loss over time.
  """
  ids = np.round(np.linspace(1,len(X),n_bins+1)+1e-10)-1
  ids = ids.astype(np.uint8)

  D_bins = [X[ids[i]:ids[i+1]+1] for i in range(0,4)]

  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    D = np.nanmean(D_bins[0])-np.mean(D_bins[3])
  return D


def std(X):
  """
  Compute standard deviation.
  """
  return np.std(X)


segmentation_statistics = {
      'decay' : decay,
      'mean'  : mean,
      'recall': recall,
      'std'   : std
      }


""" 
Utilities for computing, reading and saving benchmark evaluation.
Compute Jaccard Index. 
"""

def seg2bmap(seg, width=None, height=None):
	"""
	From a segmentation, compute a binary boundary map with 1 pixel wide
	boundaries.  The boundary pixels are offset by 1/2 pixel towards the
	origin from the actual segment boundary.

	Arguments:
		seg     : Segments labeled from 1..k.
		width	  :	Width of desired bmap  <= seg.shape[1]
		height  :	Height of desired bmap <= seg.shape[0]

	Returns:
		bmap (ndarray):	Binary boundary map.

	 David Martin <dmartin@eecs.berkeley.edu>
	 January 2003
 """

	seg = seg.astype(np.bool)
	seg[seg>0] = 1

	assert np.atleast_3d(seg).shape[2] == 1

	width  = seg.shape[1] if width  is None else width
	height = seg.shape[0] if height is None else height

	h,w = seg.shape[:2]

	ar1 = float(width) / float(height)
	ar2 = float(w) / float(h)

	assert not (width>w | height>h | abs(ar1-ar2)>0.01),\
			'Can''t convert %dx%d seg to %dx%d bmap.'%(w,h,width,height)

	e  = np.zeros_like(seg)
	s  = np.zeros_like(seg)
	se = np.zeros_like(seg)

	e[:,:-1]    = seg[:,1:]
	s[:-1,:]    = seg[1:,:]
	se[:-1,:-1] = seg[1:,1:]

	b        = seg^e | seg^s | seg^se
	b[-1,:]  = seg[-1,:]^e[-1,:]
	b[:,-1]  = seg[:,-1]^s[:,-1]
	b[-1,-1] = 0

	if w == width and h == height:
		bmap = b
	else:
		bmap = np.zeros((height,width))
		for x in range(w):
			for y in range(h):
				if b[y,x]:
					j = 1+np.floor((y-1)+height / h)
					i = 1+np.floor((x-1)+width  / h)
					bmap[j,i] = 1

	return bmap


def db_eval_boundary(foreground_mask, gt_mask, bound_th=0.008):
	"""
	Compute mean,recall and decay from per-frame evaluation.
	Calculates precision/recall for boundaries between foreground_mask and
	gt_mask using morphological operators to speed it up.

	Arguments:
		foreground_mask (ndarray): binary segmentation image.
		gt_mask         (ndarray): binary annotated image.

	Returns:
		F (float): boundaries F-measure
		P (float): boundaries precision
		R (float): boundaries recall
	"""
	assert np.atleast_3d(foreground_mask).shape[2] == 1

	bound_pix = bound_th if bound_th >= 1 else \
			np.ceil(bound_th*np.linalg.norm(foreground_mask.shape))

	# Get the pixel boundaries of both masks
	fg_boundary = seg2bmap(foreground_mask)
	gt_boundary = seg2bmap(gt_mask)
	fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
	gt_dil = binary_dilation(gt_boundary, disk(bound_pix))

	# Get the intersection
	gt_match = gt_boundary * fg_dil
	fg_match = fg_boundary * gt_dil

	# Area of the intersection
	n_fg     = np.sum(fg_boundary)
	n_gt     = np.sum(gt_boundary)

	#% Compute precision and recall
	if n_fg == 0 and  n_gt > 0:
		precision = 1
		recall = 0
	elif n_fg > 0 and n_gt == 0:
		precision = 0
		recall = 1
	elif n_fg == 0  and n_gt == 0:
		precision = 1
		recall = 1
	else:
		precision = np.sum(fg_match)/float(n_fg)
		recall    = np.sum(gt_match)/float(n_gt)

	# Compute F measure
	if precision + recall == 0:
		F = 0
	else:
		F = 2*precision*recall/(precision+recall)
	return F


def db_eval_iou(annotation, segmentation):
    """ 
    Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.

    Return:
        jaccard (float): region similarity
    """
    annotation   = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)
    if np.isclose(np.sum(annotation), 0) and np.isclose(np.sum(segmentation), 0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / \
                np.sum((annotation | segmentation),dtype=np.float32)
	

segmentation_db_measures = {
        'J': db_eval_iou,
        'F': db_eval_boundary,
        'T': db_eval_boundary
        }


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

