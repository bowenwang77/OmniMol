from functools import lru_cache

import ogb
import numpy as np
import torch
from torch.nn import functional as F
from fairseq.data import data_utils, FairseqDataset, BaseWrapperDataset
from random import Random

from .wrapper import MyPygGraphPropPredDataset
from .collator import collator

from typing import Optional, Union
from torch_geometric.data import Data as PYGDataset
from dgl.data import DGLDataset
from .dgl_datasets import DGLDatasetLookupTable, GraphormerDGLDataset
from .pyg_datasets import PYGDatasetLookupTable, GraphormerPYGDataset
from .ogb_datasets import OGBDatasetLookupTable


class BatchedDataDataset(FairseqDataset):
    def __init__(
        self, dataset, max_node=128, multi_hop_max_dist=5, spatial_pos_max=1024
    ):
        super().__init__()
        self.dataset = dataset
        self.max_node = max_node
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max

    def __getitem__(self, index):
        item = self.dataset[int(index)]
        return item

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return collator(
            samples,
            max_node=self.max_node,
            multi_hop_max_dist=self.multi_hop_max_dist,
            spatial_pos_max=self.spatial_pos_max,
        )


class TargetDataset(FairseqDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return self.dataset[index].y

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return torch.stack(samples, dim=0)


class GraphormerDataset:
    def __init__(
        self,
        dataset: Optional[Union[PYGDataset, DGLDataset]] = None,
        dataset_spec: Optional[str] = None,
        dataset_source: Optional[str] = None,
        seed: int = 0,
        train_idx = None,
        valid_idx = None,
        test_idx = None,
    ):
        super().__init__()
        if dataset is not None:
            if dataset_source == "dgl":
                self.dataset = GraphormerDGLDataset(dataset, seed=seed, train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)
            elif dataset_source == "pyg":
                self.dataset = GraphormerPYGDataset(dataset, train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)
            else:
                raise ValueError("customized dataset can only have source pyg or dgl")
        elif dataset_source == "dgl":
            self.dataset = DGLDatasetLookupTable.GetDGLDataset(dataset_spec, seed=seed)
        elif dataset_source == "pyg":
            self.dataset = PYGDatasetLookupTable.GetPYGDataset(dataset_spec, seed=seed)
        elif dataset_source == "ogb":
            self.dataset = OGBDatasetLookupTable.GetOGBDataset(dataset_spec, seed=seed)
        self.setup()

    def setup(self):
        self.train_idx = self.dataset.train_idx
        self.valid_idx = self.dataset.valid_idx
        self.test_idx = self.dataset.test_idx

        self.dataset_train = self.dataset.train_data
        self.dataset_val = self.dataset.valid_data
        self.dataset_test = self.dataset.test_data


class EpochShuffleDataset(BaseWrapperDataset):
    def __init__(self, dataset, num_samples, seed, drop_tail, batch_size):
        super().__init__(dataset)
        self.num_samples = num_samples
        self.seed = seed
        self.batch_size = batch_size
        self.drop_tail = drop_tail
        self.set_epoch(1)

    def set_epoch(self, epoch):
        with data_utils.numpy_seed(self.seed + epoch - 1):
            self.sort_order = np.arange(self.num_samples)
            if self.drop_tail:
                remain = len(self.sort_order)%self.batch_size
                self.sort_order = np.append(self.sort_order, self.sort_order[-(self.batch_size-remain):])
            self.sort_order = np.random.permutation(self.sort_order)

    def ordered_indices(self):
        return self.sort_order

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False

class BalanceShuffleDataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, batch_size, subset_length_list, resampled_sizes,dataset_isreg,drop_tail = False):
        super().__init__(dataset)
        self.seed = seed
        self.batch_size = batch_size
        self.subset_length_list = subset_length_list
        self.resampled_sizes = resampled_sizes
        self.dataset_isreg = dataset_isreg
        self.drop_tail = drop_tail
        self.set_epoch(1)

    def set_epoch(self, epoch):

        rng = Random(self.seed+epoch-1)
        # Step 1: Expand the list of lists
        expanded_list = []
        expanded_list_tags = []
        start = 0
        for idx, length in enumerate(self.subset_length_list):
            expanded_list.append(list(range(start, start + length)))
            expanded_list_tags.append([self.dataset_isreg[idx]]*length)
            start += length

        balanced_list = []
        balanced_list_tags =[]
        for i,sublist in enumerate(expanded_list):
            balanced_num = self.resampled_sizes[i]
            balanced_list.append(rng.choices(sublist,k= balanced_num ))
            balanced_list_tags.append([expanded_list_tags[i][0]]*balanced_num)

        self.sort_order = np.concatenate(balanced_list).flatten()
        if self.drop_tail:
            remain = len(self.sort_order)%self.batch_size
            self.sort_order = np.append(self.sort_order, self.sort_order[-(self.batch_size-remain):])
        with data_utils.numpy_seed(self.seed + epoch - 1):
            self.sort_order = np.random.permutation(self.sort_order)

    def ordered_indices(self):
        return self.sort_order

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False

class BalanceShuffleDataset_batch_reorganized(BaseWrapperDataset):
    def __init__(self, dataset, seed, batch_size, subset_length_list,dataset_isreg, subset_weight=0, average_sample_factor=[],drop_tail = False, distributed_world_size = 1):
        super().__init__(dataset)
        self.seed = seed
        self.batch_size = batch_size
        self.subset_length_list = subset_length_list
        self.subset_weight = subset_weight
        self.average_sample_factor = average_sample_factor
        self.drop_tail = drop_tail
        self.distributed_world_size = distributed_world_size
        self.dataset_isreg = dataset_isreg

        self.set_epoch(1)

    def set_epoch(self, epoch):

        rng = Random(self.seed+epoch-1)
        # Step 1: Expand the list of lists
        expanded_list = []
        expanded_list_tags = []
        start = 0
        for idx, length in enumerate(self.subset_length_list):
            expanded_list.append(list(range(start, start + length)))
            expanded_list_tags.append([self.dataset_isreg[idx]]*length)
            start += length
        average_length = np.array(self.subset_length_list).mean()

        if self.subset_weight>0:
            balanced_list = []
            balanced_list_tags =[]
            for i,sublist in enumerate(expanded_list):
                balanced_num = int(self.subset_weight*average_length+(1-self.subset_weight)*self.subset_length_list[i])
                balanced_list.append(rng.choices(sublist,k= balanced_num ))
                balanced_list_tags.append([expanded_list_tags[i][0]]*balanced_num)
            expanded_list = balanced_list
            expanded_list_tags = expanded_list_tags

        # Step 2: Split each sublist by self.batch_size
        split_list = []
        split_list_tags = []
        for idx,sublist in enumerate(expanded_list):
            split_list += [sublist[i:i+self.batch_size] for i in range(0, len(sublist), self.batch_size)]
            split_list_tags += [expanded_list_tags[idx][i] for i in range(0, len(sublist), self.batch_size)]
        

        # Step 3: Order the list by length, equal to batch_size first, not equal later
        split_list_reg = [idx for idx, is_reg in zip(split_list, split_list_tags) if is_reg]
        split_list_cls = [idx for idx, is_reg in zip(split_list, split_list_tags) if not is_reg]

        equal_list_reg = [i for i in split_list_reg if len(i) == self.batch_size]
        not_equal_list_reg = [i for i in split_list_reg if len(i) != self.batch_size]

        equal_list_cls = [i for i in split_list_cls if len(i) == self.batch_size]
        not_equal_list_cls = [i for i in split_list_cls if len(i) != self.batch_size]

        # equal_list_tags = [i for i in split_list_tags if len(i) == self.batch_size]
        # not_equal_list_tags = [i for i in split_list_tags if len(i) != self.batch_size]

        # Step 4: Shuffle the order of subsets that have the length equal to self.batch_size
        # import pdb
        # pdb.set_trace()
        rng.shuffle(equal_list_reg)
        rng.shuffle(equal_list_cls)
        # Combine and flatten into a single numpy array
        if self.drop_tail:
            tail_idx_reg = len(equal_list_reg)%self.distributed_world_size
            print("dropping_tail",tail_idx_reg)
            if tail_idx_reg!=0:
                final_list_reg = equal_list_reg[:-tail_idx_reg]
            else:
                final_list_reg = equal_list_reg
            tail_idx_cls = len(equal_list_cls)%self.distributed_world_size
            if tail_idx_cls!=0:
                final_list_cls = equal_list_cls[:-tail_idx_cls]
            else:
                final_list_cls = equal_list_cls
            

            grouped_final_list_reg = [final_list_reg[i:i+self.distributed_world_size] for i in range(0, len(final_list_reg), self.distributed_world_size)]
            grouped_final_list_cls = [final_list_cls[i:i+self.distributed_world_size] for i in range(0, len(final_list_cls), self.distributed_world_size)]
            final_list = grouped_final_list_reg+grouped_final_list_cls
            # rng.shuffle(final_list)

        else:
            final_list = equal_list_reg + not_equal_list_reg +equal_list_cls+not_equal_list_cls
        self.sort_order = np.concatenate(final_list).flatten()

    def ordered_indices(self):
        return self.sort_order

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False