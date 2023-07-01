import time
import torch
import numpy as np
import typing as tp


def topk_numpy(
    array: np.array,
    k: int,
) -> np.array:

    '''
    Return boolean mask (numpy indexing) for
    top k elements in input array
    '''
    
    n = len(array)

    assert n >= 0, f'Expected k > 0, got k={k}.'
    
    if n < k:

        mask = np.ones(
            shape=(n, ),
            dtype=bool,
        )

    else:
        
        mask = np.in1d(
            ar1=np.arange(n),
            ar2=np.argsort(-array)[:k],
        )
    
    return mask


def topk_torch(
    array: torch.Tensor,
    k: int,
) -> torch.Tensor:

    '''
    Return boolean mask (torch indexing) for
    top k elements in input array
    '''
    
    n = len(array)

    assert k >= 0, f'Expected k > 0, got k={k}.'

    mask = torch.zeros(
        (n, ),
        dtype=torch.bool,
        device=array.device,
    )
    
    if n < k:

        torch.bitwise_not(
            mask,
            out=mask,
        )

    else:
        
        _, indexes = torch.topk(
            array,
            k=k,
        )

        mask.scatter_(
            dim=0,
            index=indexes,
            value=torch.tensor(1),
        )
    
    return mask


def mask_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    """
    Inputs:
    mask1: NxHxW torch.float32. Consists of [0, 1]
    mask2: NxHxW torch.float32. Consists of [0, 1]
    Outputs:
    ret: NxM torch.float32. Consists of [0 - 1]
    """

    N, H, W = mask1.shape
    M, H, W = mask2.shape

    mask1 = mask1.view(N, H*W)
    mask2 = mask2.view(M, H*W)

    intersection = torch.matmul(mask1, mask2.t())

    area1 = mask1.sum(dim=1).view(1, -1)
    area2 = mask2.sum(dim=1).view(1, -1)

    union = (area1.t() + area2) - intersection

    ret = torch.where(
        union == 0,
        torch.tensor(0., device=mask1.device),
        intersection / union,
    )

    return ret


def center_mass(
    masks: torch.Tensor,
) -> torch.Tensor:

    '''
    Compute center mass in pixels for N masks
    Inputs:
    masks: torch.Tensor. Shape: NxHxW.
    Outputs:
    center_mass: torch.Tensor. Shape: Nx2.
    '''

    N = masks.shape[0]
    
    center_mass = torch.empty(
        size=(N, 2),
        dtype=torch.float32,
        device=masks.device,
    )

    for i in range(N):
        
        xy = masks[i].nonzero().to(torch.float32)
        
        center_mass[i, 0] = xy[:, 0].mean()
        center_mass[i, 1] = xy[:, 1].mean()
        
    return center_mass


class Meter:

    def __init__(
        self,
        agg_func: tp.Callable=lambda x, y: x - y,
    ) -> None:
        
        self.agg_func = agg_func
        self.T = None
        self.N = 0
        self.T_sum = 0
        self.T2_sum = 0
        self.T_min = np.inf
        self.T_max = 0
    
    def start(
        self,
        data: float,
    ) -> None:

        self.T = data
    
    def stop(
        self,
        data: float,
    ) -> None:

        T = self.agg_func(data, self.T)

        self.N += 1
        self.T_sum += T
        self.T2_sum += T**2
        self.T_min = min(T, self.T_min)
        self.T_max = max(T, self.T_max)

    def get_stats(
        self,
    ) -> list:

        avg = self.T_sum / self.N
        std = np.sqrt(self.T2_sum / self.N - avg**2)

        stats = [avg, std, self.T_min, self.T_max]
        
        return stats
