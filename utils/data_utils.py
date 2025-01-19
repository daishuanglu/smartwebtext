import os
import torch


def bin_search_left_bound(snums, target):
    left, right = 0, len(snums) - 1
    while left <= right:
        mid = (left + right) // 2
        if snums[mid] < target:
            left = mid + 1
        elif snums[mid] > target:
            right = mid - 1
        elif snums[mid] == target:
            right = mid - 1
    return right


def collate_dict(samples):
    result = {}
    for sample in samples:
        for key in sample:
            result.setdefault(key, []).append(sample[key])
    for key in result:
        if isinstance(result[key][0], float):
            result[key] = torch.FloatTensor(result[key])
    return result
