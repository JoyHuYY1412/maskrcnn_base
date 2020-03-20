# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Code is copy-pasted exactly as in torch.utils.data.distributed.
# FIXME remove this once c10d fixes the bug it has
import math
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import json
import numpy as np

class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch



class RepeatFactorTrainingDistributedSampler(Sampler):
    """
    Similar to TrainingSampler, but suitable for training on class imbalanced datasets
    like LVIS. In each epoch, an image may appear multiple times based on its "repeat
    factor". The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1] is defined
    as the fraction of images in the training set (without repeats) in which category c
    appears.
    See https://arxiv.org/abs/1908.03195 (>= v2) Appendix B.2.
    """
    # repeat_thresh:cfg.DATALOADER.REPEAT_THRESHOLD
    def __init__(self, dataset, repeat_thresh=0.001, num_replicas=None, rank=None, shuffle=True):
        """
        Args:
            dataset_dicts (list[dict]): annotations in Detectron2 dataset format.
            repeat_thresh (float): frequency threshold below which data is repeated.
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        # Get fractional repeat factors and split into whole number (_int_part)
        # and fractional (_frac_part) parts.
        self.rep_factors = self._get_repeat_factors(repeat_thresh)
        self._int_part = torch.trunc(self.rep_factors)
        self._frac_part = self.rep_factors - self._int_part

    def _get_repeat_factors(self, repeat_thresh):
        """
        Compute (fractional) per-image repeat factors.
        Args:
            See __init__.
        Returns:
            torch.Tensor: the i-th element is the repeat factor for the dataset image
                at index i.
        """

        """
        # 1. For each category c, compute the fraction of images that contain it: f(c)
        category_freq = defaultdict(int)

        for img_info in self.dataset:  # For each image (without repeats)
            cat_ids = {int(ann_id) for ann_id in img_info[1].get_field("labels")}  #tensor([id1, id2])
            print(cat_ids)
            for cat_id in cat_ids:
                category_freq[int(cat_id)] += 1
        print(category_freq[:10])
        num_images = len(self.dataset)
        for k, v in category_freq.items():
            category_freq[k] = v / num_images
        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t / f(c)))
        category_rep = {
            int(cat_id): max(1.0, math.sqrt(repeat_thresh / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }
        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        rep_factors = []
        for img_info in self.dataset:   # get repeat factor for each image
            cat_ids = {int(ann_id) for ann_id in img_info[1].get_field("labels")}
            rep_factor = max({category_rep[cat_id] for cat_id in cat_ids})
            rep_factors.append(rep_factor)
            print(rep_factors[:20])
        """


        #  too slow: load external category_info
        rep_factors_path = '/mnt/data-disk2/xinting/project/dataset/LVIS/lvis_trainval_1230/lvis_trainval_1230_repeat_factor.npy'
        rep_factors = np.load(rep_factors_path)

        return torch.tensor(rep_factors, dtype=torch.float32)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        rands = torch.rand(len(self._frac_part), generator=g)
        self.rep_factors = self._int_part + (rands < self._frac_part).float()
        indices = []
        for img_index, rep_factor in enumerate(self.rep_factors):
            indices.extend([img_index] * int(rep_factor.item()))
        if self.shuffle:
            # deterministically shuffle based on epoch
            shuffle_term = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[shuffle_term_i] for shuffle_term_i in shuffle_term]

        self.num_samples = int(math.ceil(len(indices) * 1.0 / self.num_replicas))
        total_size = self.num_samples * self.num_replicas
        # add extra samples to make it evenly divisible
        indices += indices[: (total_size - len(indices))]
        assert len(indices) == total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
