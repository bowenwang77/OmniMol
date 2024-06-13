# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
from typing import Sequence, Union
import gzip 
import json

import pickle
from functools import lru_cache
from sklearn.preprocessing import normalize
import lmdb
import random
import numpy as np
import torch
from torch import Tensor
from fairseq.data import (
    FairseqDataset,
    BaseWrapperDataset,
    NestedDictionaryDataset,
    data_utils,
)
from fairseq.tasks import FairseqTask, register_task
import os
from ..data.dataset import EpochShuffleDataset, BalanceShuffleDataset
import csv
import os

import numpy as np

# Dictionary mapping atomic symbols to atomic numbers
symbol_to_number = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96,
    'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106,
    'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116,
    'Ts': 117, 'Og': 118
}

def parse_hierarchy(db_path_list):
    """Parse the hierarchy of datasets."""
    hierarchy = {}
    for path in db_path_list:
        levels = path.split('/')
        if len(levels) == 1:
            hierarchy[levels[0]] = []
        else:
            if levels[0] not in hierarchy:
                hierarchy[levels[0]] = []
            hierarchy[levels[0]].append(levels[1])
    return hierarchy

def compute_resampled_sizes(hierarchy, subset_length_list, data_balance):
    """Compute the resampled sizes based on the hierarchy and data_balance using the corrected approach."""
    total_size = sum(subset_length_list)
    
    top_level_avg_size = total_size / len(hierarchy)
    
    resampled_sizes = []
    index = 0
    for key, value in hierarchy.items():
        if len(value) == 0:
            original_size = subset_length_list[index]
            resampled_size = (1 - data_balance) * original_size + data_balance * top_level_avg_size
            resampled_sizes.append(int(resampled_size))
            index += 1
        else:
            sub_avg_size = top_level_avg_size / len(value)
            for _ in value:
                original_size = subset_length_list[index]
                resampled_size = (1 - data_balance) * original_size + data_balance * sub_avg_size
                resampled_sizes.append(int(resampled_size))
                index += 1
    return resampled_sizes

def scale_sizes(resampled_sizes, data_scale):
    """Scale the computed sizes."""
    return [int(size * data_scale) for size in resampled_sizes]

def remove_duplicates(dir_list):
    seen = set()
    unique_dirs = []

    for dir_path in dir_list:
        dataset_name = dir_path.split('/')[-1]
        if dataset_name not in seen:
            seen.add(dataset_name)
            unique_dirs.append(dir_path)

    return unique_dirs


class LMDBDataset:
    def __init__(self, db_path):
        # db_path = db_path[:-5]+"_cleaned.lmdb"
        super().__init__()
        assert Path(db_path).exists(), f"{db_path}: No such file or directory"
        # self.env = lmdb.Environment(
        #     db_path,
        #     map_size=(1024 ** 3) * 256,
        #     subdir=False,
        #     readonly=True,
        #     readahead=True,
        #     meminit=False,
        # )
        self.env = lmdb.open(
            str(db_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        self.len: int = self.env.stat()["entries"]
        if db_path.split("/")[-3]=='pcq' and db_path.split("/")[-2]=="train":
            db_path_to_keys = db_path[:-9]+"keys.pkl"
            if not Path(db_path_to_keys).exists():
                with self.env.begin() as txn:
                    self._keys = list(txn.cursor().iternext(values=False))
                with open(db_path_to_keys, "wb") as f:
                    pickle.dump(self._keys, f)
            else:
                self._keys = pickle.load(open(db_path_to_keys,"rb"))
        else:
            with self.env.begin() as txn:
                self._keys = list(txn.cursor().iternext(values=False))

    def __len__(self):
        return self.len

    @lru_cache(maxsize=16)
    def __getitem__(self, idx: int) -> dict[str, Union[Tensor, float]]:
        if idx < 0 or idx >= self.len:
            raise IndexError
        if self.env.begin().get(f"{idx}".encode()) is None:
            key = self._keys[idx]
            datapoint_pickled = self.env.begin().get(key)
            datapoint_pickled = gzip.decompress(datapoint_pickled)
            data = pickle.loads(datapoint_pickled)
            data["id"] = int.from_bytes(key, "big")
            pcq = True
        else:
            data = pickle.loads(self.env.begin().get(f"{idx}".encode()))
            pcq = False
        if isinstance(data,dict):
            if pcq:
                return dict(
                    pos = torch.as_tensor(data["input_pos"]).float(),
                    pos_relaxed=torch.as_tensor(data["label_pos"]).float(),
                    atoms = torch.as_tensor(np.vectorize(symbol_to_number.get)(data['atoms']).astype(np.int32)).long(),
                    relaxed_energy= data["target"],
                    local_idx = data["id"],
                    sid = data['id'],
                    property_name = "pcq",
                    smi = data['smi'],
                    dataset_type = 'pcq',
                )

            if 'idx' in data.keys():
                return dict(
                    pos=torch.as_tensor(data["coordinates"]).float(),
                    atoms=torch.as_tensor(data["atoms"]).long(),
                    relaxed_energy=data["target"],  # python float
                    local_idx = data["local_idx"],
                    sid = data['idx'],
                    property_name= data['property_name'],
                    smi = data['smi'],
                    dataset_type = 'combined_ADMET'
                )
            elif 'local_idx' in data.keys():
                return dict(
                    pos=torch.as_tensor(data["coordinates"]).float(),
                    atoms=torch.as_tensor(data["atoms"]).long(),
                    relaxed_energy=data["target"],  # python float
                    local_idx = data["local_idx"],
                    sid= data["local_idx"],
                    property_name= data['property_name'],
                    smi = data['smi'],
                    dataset_type = 'single_ADMET'
                )
        else:
            return dict(
                pos=torch.as_tensor(data["pos"]).float(),
                pos_relaxed=torch.as_tensor(data["pos_relaxed"]).float(),
                cell=torch.as_tensor(data["cell"]).float().view(3, 3),
                atoms=torch.as_tensor(data["atomic_numbers"]).long(),
                tags=torch.as_tensor(data["tags"]).long(),
                relaxed_energy=data["y_relaxed"],  # python float
                sid = data["sid"],
                dataset_type = 'OCP'
            )        


class LMDBDataset_test:
    def __init__(self, db_path):
        super().__init__()
        assert Path(db_path).exists(), f"{db_path}: No such file or directory"
        # self.env = lmdb.Environment(
        #     db_path,
        #     map_size=(1024 ** 3) * 256,
        #     subdir=False,
        #     readonly=True,
        #     readahead=True,
        #     meminit=False,
        # )
        self.env = lmdb.open(
            str(db_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        self.len: int = self.env.stat()["entries"]

    def __len__(self):
        return self.len

    @lru_cache(maxsize=16)
    def __getitem__(self, idx: int) -> dict[str, Union[Tensor, float]]:
        if idx < 0 or idx >= self.len:
            raise IndexError
        data = pickle.loads(self.env.begin().get(f"{idx}".encode()))
        return dict(
            pos=torch.as_tensor(data["pos"]).float(),
            cell=torch.as_tensor(data["cell"]).float().view(3, 3),
            atoms=torch.as_tensor(data["atomic_numbers"]).long(),
            tags=torch.as_tensor(data["tags"]).long(),
            sid = torch.as_tensor(data["sid"]).long(),
        )


class PBCDataset_NoisyNodes:
    def __init__(self, dataset, dataset_dir_list, use_noisy_node: bool, noise_scale: float, 
    noise_type: str, noisy_node_rate: float, 
    noise_deltapos_normed: bool, noise_in_traj: bool,
    noisy_label: bool, noisy_label_downscale: float,
    remove_outliers: bool,task_dict: dict,
    ensembled_testing: int = -1,
    split: str="train",challenge_testing: bool = False):
        self.lists_of_dataset = False
        if isinstance(dataset,list):
            self.lists_of_dataset = True
        self.dataset = dataset
        self.dataset_dir_list = dataset_dir_list
        self.use_noisy_node = use_noisy_node
        # self.task_dict = pickle.load(open(r"/HOME/DRFormer_ADMET/graphormer/task_dictionary_mean_std_BBBP_BACE.pkl", "rb"))
        self.task_dict = task_dict       
        self.cell_offsets = torch.tensor(
            [
                [-1, -1, 0],
                [-1, 0, 0],
                [-1, 1, 0],
                [0, -1, 0],
                [0, 1, 0],
                [1, -1, 0],
                [1, 0, 0],
                [1, 1, 0],
            ],
        ).float()
        self.n_cells = self.cell_offsets.size(0)
        self.cutoff = 8
        self.filter_by_tag = True ##If true, only consider expanded nodes that have dist<cutoff with center adsorbate
        self.noise_scale = noise_scale
        self.noise_type = noise_type
        self.noisy_node_rate = noisy_node_rate
        self.noise_deltapos_normed = noise_deltapos_normed
        self.noise_in_traj = noise_in_traj
        self.noisy_label = noisy_label
        self.remove_outliers = remove_outliers
        self.noisy_label_downscale = noisy_label_downscale
        self.ensembled_testing = ensembled_testing
        self.split = split
        self.challenge_testing = challenge_testing
        # if self.noise_type == "trunc_normal":
            # self.noise_f = lambda num_mask: np.clip(
            #     np.random.randn(num_mask, 3) * self.noise_scale,
            #     a_min=-self.noise_scale * 2.0,
            #     a_max=self.noise_scale * 2.0,
            # )
        if self.noise_type == "normal":
            # self.noise_f = lambda num_mask: np.random.randn(num_mask, 3) * self.noise_scale
            self.noise_f = lambda num_mask: normalize(np.random.randn(num_mask, 3),axis=1,norm="l2") * np.random.randn(num_mask,1) * self.noise_scale
        elif self.noise_type == "uniform":
            # self.noise_f = lambda num_mask: np.random.uniform(
            #     low=-self.noise_scale, high=self.noise_scale, size=(num_mask, 3)
            # )
            self.noise_f = lambda num_mask: normalize(np.random.randn(num_mask, 3),axis=1,norm="l2") * np.random.rand(num_mask,1) * self.noise_scale
        else:
            self.noise_f = lambda num_mask: 0.0

    def __len__(self):
        if isinstance(self.dataset,list):
            num = 0
            for data_id in range(len(self.dataset)):
                num += len(self.dataset[data_id])
            return num
        return len(self.dataset)
    
    def get_item_and_sublist_index(self, list_of_lists, idx):
        current_idx = 0
        for i, sublist in enumerate(list_of_lists):
            sublist_length = len(sublist)
            if idx < current_idx + sublist_length:
                return sublist[idx - current_idx], i
            current_idx += sublist_length
        return None, -1  # Return (None, -1) if the index is out of range
    
    @lru_cache(maxsize=16)
    def __getitem__(self, idx):      
        
        if self.lists_of_dataset:
            # dataset_idx = random.randint(0,len(self.dataset)-1)

            # data = self.dataset[dataset_idx][idx]
            data,dataset_idx = self.get_item_and_sublist_index(self.dataset, idx)
            dataset_dir = self.dataset_dir_list[dataset_idx]
        else:
            data = self.dataset[idx]
            dataset_dir = self.dataset_dir_list[0]
        if data['dataset_type'] in [ 'combined_ADMET', 'single_ADMET']:
            if (self.split == "val_id" or self.split == "test_id") and self.ensembled_testing != -1:
                pos_full = data["pos"].clone()
            else:
                frame_id = random.randint(0,data["pos"].shape[0]-1)
                pos_full = data["pos"][frame_id].clone()
            atoms_full = data["atoms"]
            num_atoms = atoms_full.shape[0]
            tags_full = 2*torch.ones_like(atoms_full)
            real_mask_full = torch.ones_like(tags_full, dtype=torch.bool)
            smi = data['smi']
            cell = torch.zeros(3,3)
            property_name = data['property_name']
            local_idx = data['local_idx']
        elif data['dataset_type'] in ['pcq']:
            if (self.split == "val_id" or self.split == "test_id") and self.ensembled_testing != -1:
                pos_full = data["pos"].clone()
            else:
                frame_id = random.randint(0,data["pos"].shape[0]-1)
                pos_full = data["pos"][frame_id].clone()
            atoms_full = data["atoms"]
            num_atoms = atoms_full.shape[0]
            tags_full = 2*torch.ones_like(atoms_full)
            real_mask_full = torch.ones_like(tags_full, dtype=torch.bool)
            smi = data['smi']
            cell = torch.zeros(3,3)
            property_name = data['property_name']
            local_idx = data['local_idx']
        else:
            pos = data["pos"]
            cell = data["cell"]
            atoms = data["atoms"]
            tags = data["tags"]

            offsets = torch.matmul(self.cell_offsets, cell).view(self.n_cells, 1, 3)
            expand_pos = (pos.unsqueeze(0).expand(self.n_cells, -1, -1) + offsets).view(
                -1, 3
            )
            src_pos = pos[tags > 1] if self.filter_by_tag else pos

            dist: Tensor = (src_pos.unsqueeze(1) - expand_pos.unsqueeze(0)).norm(dim=-1)
            used_mask = (dist < self.cutoff).any(dim=0) & tags.ne(2).repeat(
                self.n_cells
            )  # not copy ads
            used_expand_pos = expand_pos[used_mask]
            used_expand_tags = tags.repeat(self.n_cells)[
                used_mask
            ]  # original implementation use zeros, need to test
            atoms_full = torch.cat([atoms, atoms.repeat(self.n_cells)[used_mask]])
            pos_full = torch.cat([pos, used_expand_pos], dim=0)        
            tags_full = torch.cat([tags, used_expand_tags])
            real_mask_full = torch.cat(
                            [
                                torch.ones_like(tags, dtype=torch.bool),
                                torch.zeros_like(used_expand_tags, dtype=torch.bool),
                            ]
                        )


            if dataset_dir.split('/')[-1] == 'SAA':
                smi = 'OCP_SAA_sid_'+str(data['sid'])
                property_name = 'OCP_SAA'
            else:
                smi = 'OCP_OC20_sid_'+str(data['sid'])
                property_name = 'OCP_OC20'
            local_idx = data['sid']

        # task_emb = self.task_dict[property_name]['embedding']
        is_reg = self.task_dict[property_name]['regression']
        cls_num = self.task_dict[property_name]['cls_num']
        task_idx = self.task_dict[property_name]['task_idx']
        task_mean = self.task_dict[property_name]['task_mean']
        task_std = self.task_dict[property_name]['task_std']
        prop_id = self.task_dict[property_name].get('prop_id',9)
        task_geo_mean = self.task_dict[property_name].get('task_geo_mean',[0.0,0.0,0.0])
        task_geo_std = self.task_dict[property_name].get('task_geo_std',[1.0,1.0,1.0])
        max_node = 200
        if pos_full.shape[0] > max_node:
            pos_full = pos_full[:max_node]
            atoms_full = atoms_full[:max_node]
            tags_full = tags_full[:max_node]
            num_atoms = max_node
            
        output_dict = dict(
                pos=pos_full,
                atoms=atoms_full,
                tags = tags_full,
                real_mask = real_mask_full, #added
                sid= data['sid'],
                cell = cell,
                local_idx = local_idx,
                smi = smi,
                property_name= property_name,
                # task_emb = task_emb,
                is_reg = is_reg,
                cls_num = cls_num,
                task_idx = task_idx,
                task_mean = task_mean,
                prop_id = prop_id,
                task_std = task_std,
                task_geo_mean = task_geo_mean,
                task_geo_std = task_geo_std,
            )

        if not (self.challenge_testing and self.split in ["test_id",
                                                            "test_ood_ads",
                                                            "test_ood_cat",
                                                            "test_ood_both"]):
            
            if data['dataset_type'] in [ 'combined_ADMET', 'single_ADMET']:
                noisy_node_mask = torch.rand(num_atoms)<self.noisy_node_rate
                label_noise = self.noise_f(noisy_node_mask.sum())
                label_noise = torch.from_numpy(label_noise).type_as(pos_full)
                deltapos_full = torch.zeros_like(pos_full)
                # print(pos_full.shape, noisy_node_mask.shape, label_noise.shape)
                if (self.split == "val_id" or self.split == "test_id") and self.ensembled_testing != -1:
                    pos_full[:,noisy_node_mask] -= label_noise
                    deltapos_full[:,noisy_node_mask] = label_noise
                else:
                    pos_full[noisy_node_mask] -= label_noise
                    deltapos_full[noisy_node_mask] = label_noise  
            elif data['dataset_type'] in ['pcq']:
                pos_relaxed = data["pos_relaxed"]
                if self.split in ['val_id','test_id']:
                    deltapos_full = torch.zeros_like(pos_full)
                else:
                    deltapos_full = pos_relaxed - pos_full
            else:
                ##Requires label
                pos_relaxed = data["pos_relaxed"]
                expand_pos_relaxed = (
                    pos_relaxed.unsqueeze(0).expand(self.n_cells, -1, -1) + offsets ###This is so wrong! Previously the source code use pos(initial)
                ).view(-1, 3)
                used_expand_pos_relaxed = expand_pos_relaxed[used_mask]
                deltapos_full=torch.cat(
                    [pos_relaxed - pos, used_expand_pos_relaxed - used_expand_pos], dim=0
                )
                if deltapos_full.shape[0] > max_node:
                    deltapos_full = deltapos_full[:max_node]
                    
                if self.use_noisy_node:
                    deltapos_norm = deltapos_full.norm(dim =1)
                    deltapos_nonzero = deltapos_norm!=0
                    noisy_node_mask = torch.logical_and(torch.rand(deltapos_full.shape[0])<self.noisy_node_rate,deltapos_nonzero)
                    # if deltapos_full.shape[0] > max_node:
                    #     noisy_node_mask = noisy_node_mask[:max_node]
                    pos_full_noise = pos_full.clone()
                    deltapos_full_noise = deltapos_full.clone()
                    pos_full_noisy_label = pos_full.clone()+deltapos_full.clone()
                    deltapos_full_noisy_label = torch.zeros_like(deltapos_full)
                    if noisy_node_mask.sum()!=0: ##in case of error instances that have initial_pos==relaxed_pos
                        node_noise = self.noise_f(noisy_node_mask.sum())
                        node_noise = torch.from_numpy(node_noise).type_as(deltapos_full)
                        if self.noise_deltapos_normed:
                                node_noise *= deltapos_norm[noisy_node_mask].unsqueeze(-1)
                                node_noise /= deltapos_norm[noisy_node_mask].unsqueeze(-1).mean()
                        if self.noise_in_traj:
                            node_noise += (deltapos_full*torch.rand(deltapos_full.shape[0]).unsqueeze(-1))[noisy_node_mask]
                        pos_full_noise[noisy_node_mask] += node_noise
                        deltapos_full_noise[noisy_node_mask] -= node_noise
                        if self.noisy_label:
                            label_noise = self.noise_f(noisy_node_mask.sum()) * self.noisy_label_downscale
                            # if deltapos_full.shape[0] > max_node:
                            #     label_noise = label_noise[:max_node]
                            label_noise = torch.from_numpy(label_noise).type_as(deltapos_full)
                            deltapos_full_noisy_label[noisy_node_mask] = label_noise
                            pos_full_noisy_label[noisy_node_mask] -= label_noise
                            # deltapos_full_noise[noisy_node_mask] += label_noise
                    if self.noisy_label:
                        idx = random.randint(0,2)
                        pos_full = [pos_full,pos_full_noise,pos_full_noisy_label][idx]
                        deltapos_full = [deltapos_full,deltapos_full_noise,deltapos_full_noisy_label][idx]
                    else:
                        idx = random.randint(0,1)
                        pos_full = [pos_full,pos_full_noise][idx]
                        deltapos_full = [deltapos_full,deltapos_full_noise][idx]      
            output_dict['relaxed_energy']=data["relaxed_energy"]
            output_dict['pos'] = pos_full  
            output_dict['deltapos']=deltapos_full

        if property_name=="rotation_strength":
            output_dict['relaxed_energy']=output_dict['relaxed_energy'][0]
            output_dict['task_mean']   =output_dict['task_mean'][0]
            output_dict['task_std'] =output_dict['task_std'][0]
        return output_dict


def pad_1d(samples: Sequence[Tensor], fill=0, multiplier=8):
    max_len = max(x.size(0) for x in samples)
    max_len = (max_len + multiplier - 1) // multiplier * multiplier
    n_samples = len(samples)
    out = torch.full(
        (n_samples, max_len, *samples[0].shape[1:]), fill, dtype=samples[0].dtype
    )
    for i in range(n_samples):
        x_len = samples[i].size(0)
        out[i][:x_len] = samples[i]
    return out

def pad_1d_ensembled(samples: Sequence[Tensor], fill=0, multiplier=8):
    max_len = max(x.size(-2) for x in samples)
    max_len = (max_len + multiplier - 1) // multiplier * multiplier
    n_samples = len(samples)
    out = torch.full(
        (n_samples,samples[0].shape[0], max_len, *samples[0].shape[2:]), fill, dtype=samples[0].dtype
    )
    for i in range(n_samples):
        x_len = samples[i].size(-2)
        out[i][:,:x_len,:] = samples[i]
    return out


class AtomDataset(FairseqDataset):
    def __init__(self, dataset, keyword):
        super().__init__()
        self.dataset = dataset
        self.keyword = keyword
        self.atom_list = [
            1,
            5,
            6,
            7,
            8,
            11,
            13,
            14,
            15,
            16,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            55,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
        ]
        # fill others as unk
        unk_idx = len(self.atom_list) + 1
        self.atom_mapper = torch.full((128,), unk_idx)
        for idx, atom in enumerate(self.atom_list):
            self.atom_mapper[atom] = idx + 1  # reserve 0 for paddin

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        atoms: Tensor = self.dataset[index][self.keyword]
        return self.atom_mapper[atoms]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return pad_1d(samples)


class KeywordDataset(FairseqDataset):
    def __init__(self, dataset, keyword, is_scalar=False,is_scalar_list = False, ensembled_testing=-1, pad_fill=0, is_str = False):
        super().__init__()
        self.dataset = dataset
        self.keyword = keyword
        self.is_scalar = is_scalar
        self.is_scalar_list = is_scalar_list
        self.pad_fill = pad_fill
        self.ensembled_testing = ensembled_testing
        self.is_str = is_str

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return self.dataset[index][self.keyword]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if self.is_str: 
            return samples
        elif self.is_scalar:
            return torch.tensor(samples)
        elif self.is_scalar_list:
            return torch.tensor(samples).unsqueeze(-1)
        elif (self.dataset.split == "val_id" or self.dataset.split == "test_id") and self.ensembled_testing != -1:
            return pad_1d_ensembled(samples, fill=self.pad_fill)
        return pad_1d(samples, fill=self.pad_fill)


@register_task("dft_md_combine")
class DFT_MD_COMBINE(FairseqTask):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument("data", metavar="FILE", help="directory for data")

    @property
    def target_dictionary(self):
        return None

    def load_dataset(self, split, combine=False, **kwargs):
        assert split in [
            "train",
            "val_id",
            "val_ood_ads",
            "val_ood_cat",
            "val_ood_both",
            "test_id",
            "test_ood_ads",
            "test_ood_cat",
            "test_ood_both",
        ], "invalid split: {}!".format(split)
        print(" > Loading {} ...".format(split))

        #Not considering SAA dataset condition and full dataset
        if self.cfg.mixing_dataset:
            dataset_dir_list = []
            for subdir in os.listdir(self.cfg.data):
                if os.path.isdir(os.path.join(self.cfg.data, subdir)):
                    if subdir.split('_')[0] == 'ADMET':
                        subsubdir_list = os.listdir(os.path.join(self.cfg.data, subdir))
                        for subsubdir in subsubdir_list:
                            dataset_dir_list.append(os.path.join(subdir,subsubdir))
                    else:
                        dataset_dir_list.append(subdir)
            dataset_dir_list = remove_duplicates(dataset_dir_list)
            dataset_dir_list = sorted(dataset_dir_list)
            # dataset_dir_list = [os.path.join(self.cfg.data, subdir) for subdir in os.listdir(self.cfg.data) if os.path.isdir(os.path.join(self.cfg.data, subdir))]
            db_path_list = [str(Path(subdir) / split / "data.lmdb") for subdir in dataset_dir_list] 
            #Order the db_path_list
            # db_path_list = sorted(db_path_list)
            lmdb_dataset = [LMDBDataset(os.path.join(self.cfg.data,db_path)) for db_path in db_path_list]
            subset_length_list = [len(subset) for subset in lmdb_dataset]
        else:
            dataset_dir_list = [self.cfg.data]
            db_path = str(Path(self.cfg.data) / split / "data.lmdb")
            lmdb_dataset = [LMDBDataset(db_path)]
            subset_length_list = [len(lmdb_dataset)]
        # import pdb
        # pdb.set_trace()
        if self.cfg.task_dict_dir.split(".")[-1] == "pkl":
            task_dict = pickle.load(open(self.cfg.task_dict_dir, "rb")) 
        else:
            with open(self.cfg.task_dict_dir, 'r') as file:
                task_dict = json.load(file)

        dataset_isreg = []
        # import pdb
        # pdb.set_trace()
        # print("length",len(lmdb_dataset))
        for idx in range(len(lmdb_dataset)):
            # print(idx)
            # print(lmdb_dataset[idx])
            # print(lmdb_dataset[idx][0])
            # print(lmdb_dataset[idx][0]['dataset_type'])
            # print()
            if lmdb_dataset[idx][0]['dataset_type'] in [ 'combined_ADMET', 'single_ADMET']:
                property_name = lmdb_dataset[idx][0]['property_name']
                print(property_name)
                dataset_isreg.append(task_dict[property_name]['regression'])
            else:
                dataset_isreg.append(True)


        use_noisy_nodes = self.cfg.noisy_nodes and split == "train"
        noisy_label = self.cfg.noisy_label and use_noisy_nodes
        pbc_dataset = PBCDataset_NoisyNodes(lmdb_dataset,dataset_dir_list,use_noisy_nodes,self.cfg.noise_scale,
                self.cfg.noise_type, self.cfg.noisy_nodes_rate,  
                self.cfg.noise_deltapos_normed, self.cfg.noise_in_traj, 
                noisy_label, self.cfg.noisy_label_downscale,
                self.cfg.remove_outliers, task_dict, 
                self.cfg.ensembled_testing,split,self.cfg.challenge_testing)
        
        atoms = AtomDataset(pbc_dataset, "atoms")
        tags = KeywordDataset(pbc_dataset, "tags")
        real_mask = KeywordDataset(pbc_dataset, "real_mask")
        sid = KeywordDataset(pbc_dataset, "sid", is_scalar=True)
        local_idx = KeywordDataset(pbc_dataset, "local_idx", is_scalar=True)
        cell = KeywordDataset(pbc_dataset, "cell")
        ensembled_testing = -1
        if split == "val_id" or split == "test_id":
            ensembled_testing = self.cfg.ensembled_testing
            pos = KeywordDataset(pbc_dataset, "pos", ensembled_testing=ensembled_testing)
        else:
            pos = KeywordDataset(pbc_dataset, "pos")
        # task_emb = KeywordDataset(pbc_dataset, "task_emb")
        is_reg = KeywordDataset(pbc_dataset, "is_reg", is_scalar=True)
        task_mean = KeywordDataset(pbc_dataset, "task_mean", is_scalar=True)
        task_std = KeywordDataset(pbc_dataset, "task_std", is_scalar=True)
        prop_id = KeywordDataset(pbc_dataset, "prop_id", is_scalar=True)
        task_geo_mean = KeywordDataset(pbc_dataset, "task_geo_mean", is_scalar=True)
        task_geo_std = KeywordDataset(pbc_dataset, "task_geo_std", is_scalar=True)
        cls_num = KeywordDataset(pbc_dataset, "cls_num", is_scalar=True)
        task_idx = KeywordDataset(pbc_dataset, "task_idx", is_scalar=True)

        smi = KeywordDataset(pbc_dataset, "smi", is_str=True)
        property_name = KeywordDataset(pbc_dataset, "property_name", is_str=True)

        if not (self.cfg.challenge_testing and split in ["test_id",
                                                            "test_ood_ads",
                                                            "test_ood_cat",
                                                            "test_ood_both"]):
            relaxed_energy = KeywordDataset(pbc_dataset, "relaxed_energy", is_scalar=True)
            if split == "val_id" or split == "test_id":
                ensembled_testing = self.cfg.ensembled_testing
                deltapos = KeywordDataset(pbc_dataset, "deltapos", ensembled_testing=ensembled_testing)
            else:
                deltapos = KeywordDataset(pbc_dataset, "deltapos")
            dataset = NestedDictionaryDataset(
                {
                    "net_input": {
                        "pos": pos,
                        "atoms": atoms,
                        "tags": tags,
                        "real_mask": real_mask,
                        # "task_emb": task_emb,
                        "task_idx": task_idx,
                        "is_reg": is_reg,
                        "cls_num":cls_num,
                    },
                    "task_input": {
                        "sid": sid,
                        "cell": cell,
                        "local_idx": local_idx,
                        "task_mean": task_mean,
                        "task_std": task_std,
                        "prop_id": prop_id,
                        "task_geo_mean":task_geo_mean,
                        "task_geo_std":task_geo_std,
                        "smi": smi,
                        "property_name": property_name,
                    },
                    "targets": {
                        "relaxed_energy": relaxed_energy,
                        "deltapos": deltapos,
                    }
                },
                sizes=[np.zeros(len(atoms))],
            )
        else:
            dataset = NestedDictionaryDataset(
                {
                    "net_input": {
                        "pos": pos,
                        "atoms": atoms,
                        "tags": tags,
                        "real_mask": real_mask,
                        # "task_emb": task_emb,
                        "task_idx": task_idx,
                        "is_reg": is_reg,
                        "cls_num":cls_num,
                    },
                    "task_input": {
                        "sid": sid,
                        "cell": cell,
                        "local_idx": local_idx,
                        "task_mean": task_mean,
                        "task_std": task_std,
                        "prop_id": prop_id,
                        "task_geo_mean":task_geo_mean,
                        "task_geo_std":task_geo_std,
                        "smi": smi,
                        "property_name": property_name,
                    },
                },
                sizes=[np.zeros(len(atoms))],
            )
        resampled_sizes = subset_length_list
        if split == "train":
            if self.cfg.data_balance>0:
                hierarchy = parse_hierarchy(db_path_list)
                resampled_sizes = compute_resampled_sizes(hierarchy, subset_length_list, self.cfg.data_balance)
                resampled_sizes = scale_sizes(resampled_sizes, self.cfg.data_scale)
                dataset = BalanceShuffleDataset(
                    dataset,
                    seed = self.cfg.seed,
                    batch_size = self.cfg.batch_size,
                    subset_length_list = subset_length_list,
                    resampled_sizes = resampled_sizes,
                    dataset_isreg = dataset_isreg,
                    drop_tail= self.cfg.drop_tail,
                )
            else:
                dataset = EpochShuffleDataset(
                    dataset,
                    num_samples = len(atoms),
                    seed = self.cfg.seed,
                    drop_tail= self.cfg.drop_tail,
                    batch_size = self.cfg.batch_size,
                )
        else:
            if self.cfg.drop_tail:
                dataset = EpochShuffleDataset(
                    dataset,
                    num_samples = len(atoms),
                    seed = self.cfg.seed,
                    drop_tail= self.cfg.drop_tail,
                    batch_size = self.cfg.batch_size,
                )
        # Saving dataset statistics
        # Define the directory where to save the .CSV file
        save_dir = self.cfg.save_dir
        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
        # Define the file path for the .CSV file based on the split name
        csv_file_path = os.path.join(save_dir, f"{split}_dataset_statistics.csv")
        
        print("| Loaded {} with {} samples".format(split, len(dataset)))
        # Data to be written to the CSV file
        data_to_write = []

        # Collect data
        if self.cfg.mixing_dataset:
            for dataset_idx in range(len(lmdb_dataset)):
                dataset_name = db_path_list[dataset_idx].split("/")[-3]
                dataset_length = subset_length_list[dataset_idx]
                sampled_to = resampled_sizes[dataset_idx]
                print("| {} has {} raw datapoint, sampled to {}".format(dataset_name, dataset_length, sampled_to))
                data_to_write.append([dataset_name, dataset_length, sampled_to])
        self.datasets[split] = dataset

        # Writing to CSV
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Writing headers
            writer.writerow(["Dataset Name", "Raw Data Points", "Sampled To"])
            # Writing the data
            for row in data_to_write:
                writer.writerow(row)

        print("CSV file saved at:", csv_file_path)