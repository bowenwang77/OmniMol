#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""
import scipy
import sklearn
import sklearn.metrics
from torch import nn
import pandas as pd
import argparse
import logging
import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple
import wandb
import time
# We need to setup root logger before importing any fairseq libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from fairseq import checkpoint_utils, options, quantization_utils, tasks, utils
from fairseq.data import data_utils, iterators
from fairseq.data.plasma_utils import PlasmaStore
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap
from fairseq.distributed import utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer
from filelock import FileLock

def main(cfg: FairseqConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    if (
        distributed_utils.is_master(cfg.distributed_training)
        and "job_logging_cfg" in cfg
    ):
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    if cfg.common.log_file is not None:
        handler = logging.FileHandler(filename=cfg.common.log_file)
        logger.addHandler(handler)

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    logger.info(cfg)

    if cfg.checkpoint.write_checkpoints_asynchronously:
        try:
            import iopath  # noqa: F401
        except ImportError:
            logging.exception(
                "Asynchronous checkpoint writing is specified but iopath is "
                "not installed: `pip install iopath`"
            )
            return

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)

    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    if cfg.distributed_training.ddp_backend == "fully_sharded":
        with fsdp_enable_wrap(cfg.distributed_training):
            model = fsdp_wrap(task.build_model(cfg.model))
    else:
        model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)
    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(
                p.numel() for p in model.parameters() if not getattr(p, "expert", False)
            ),
            sum(
                p.numel()
                for p in model.parameters()
                if not getattr(p, "expert", False) and p.requires_grad
            ),
        )
    )

    logger.info(
        "num. expert model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False)),
            sum(
                p.numel()
                for p in model.parameters()
                if getattr(p, "expert", False) and p.requires_grad
            ),
        )
    )

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    # We load the valid dataset AFTER building the model
    data_utils.raise_if_valid_subsets_unintentionally_ignored(cfg)
    if cfg.dataset.combine_valid_subsets:
        task.load_dataset("valid", combine=True, epoch=1)
    else:
        for valid_sub_split in cfg.dataset.valid_subset.split(","):
            task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # (optionally) Configure quantization
    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if cfg.common.model_parallel_size == 1:
        trainer = Trainer(cfg, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(cfg, task, model, criterion)
    logger.info(
        "training on {} devices (GPUs/TPUs)".format(
            cfg.distributed_training.distributed_world_size
        )
    )
    logger.info(
        "max tokens per device = {} and max sentences per device = {}".format(
            cfg.dataset.max_tokens,
            cfg.dataset.batch_size,
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )
    if cfg.common.tpu:
        import torch_xla.core.xla_model as xm

        xm.rendezvous("load_checkpoint")  # wait for all workers

    max_epoch = cfg.optimization.max_epoch or math.inf
    lr = trainer.get_lr()

    train_meter = meters.StopwatchMeter()
    train_meter.start()
    # if cfg.dataset.valid_subset.split(",")[0].split("_")[0]=='test' and cfg.task.data.split('/')[-1]!='SAA' and cfg.task._name.split('_')[0]!='admet':
    if cfg.model.challenge_testing and cfg.dataset.valid_subset.split(",")[0].split("_")[0]=='test':
        test(cfg, trainer, task, epoch_itr)
    while epoch_itr.next_epoch_idx <= max_epoch:
        if lr <= cfg.optimization.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr}) is smaller "
                "than or equal to minimum learning rate "
                f"(--stop-min-lr={cfg.optimization.stop_min_lr})"
            )
            break

        # train for one epoch
        valid_losses, should_stop = train(cfg, trainer, task, epoch_itr)
        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
        )
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))
    # os.system("pkill python")
    # ioPath implementation to wait for all asynchronous file writes to complete.
    if cfg.checkpoint.write_checkpoints_asynchronously:
        logger.info(
            "ioPath PathManager waiting for all asynchronous checkpoint "
            "writes to finish."
        )
        PathManager.async_close()
        logger.info("ioPath PathManager finished waiting.")


def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    cfg.checkpoint.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("test")
def test(
    cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    itr = iterators.GroupedIterator(
        itr,
        update_freq,
        skip_remainder_batch=cfg.optimization.skip_remainder_batch,
    )
    if cfg.common.tpu:
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_file=cfg.common.log_file,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        wandb_project=(
            cfg.common.wandb_project
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
        azureml_logging=(
            cfg.common.azureml_logging
            if distributed_utils.is_master(cfg.distributed_training)
            else False
        ),
    )
    progress.update_config(_flatten_config(cfg))

    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = cfg.dataset.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    logger.info("Start iterating over samples")

    for i, samples in enumerate(progress):
        # with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
        #     "train_step-%d" % i
        # ):  
        #     log_output = trainer.train_step(samples) #loss is summation (over samples) of recovered(* std) MAE

        # if log_output is not None:  # not OOM, overflow, ...
        #     # log mid-epoch stats
        #     num_updates = trainer.get_num_updates()
        #     if num_updates % cfg.common.log_interval == 0:
        #         stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
        #         progress.log(stats, tag="train_inner", step=num_updates)

        #         # reset mid-epoch stats after each log interval
        #         # the end-of-epoch stats will still be preserved
        #         metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        test_and_save(
            cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


@metrics.aggregate("train")
def train(
    cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    itr = iterators.GroupedIterator(
        itr,
        update_freq,
        skip_remainder_batch=cfg.optimization.skip_remainder_batch,
    )
    if cfg.common.tpu:
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_file=cfg.common.log_file,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        wandb_project=(
            cfg.common.wandb_project
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
        azureml_logging=(
            cfg.common.azureml_logging
            if distributed_utils.is_master(cfg.distributed_training)
            else False
        ),
        sweep_mode = cfg.model.sweep_mode
    )
    progress.update_config(_flatten_config(cfg))

    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = cfg.dataset.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    logger.info("Start iterating over samples")

    # rec_node_label_full = []
    # rec_node_label_center = []
    # rec_edge_type_tags = []
    # rec_edge_labels = []
    # for i,samples in enumerate(progress):
        
    #     pos = samples[0]['net_input']['pos']

    #     tags = samples[0]['net_input']['tags']+1
    #     deltapos_node = samples[0]["targets"]["deltapos"]
    #     non_padding_mask = samples[0]["net_input"]["atoms"].ne(0)
    #     real_mask = samples[0]['net_input']["real_mask"]
    #     center_mask = (samples[0]['net_input']['tags']>0) & real_mask
    #     tags_full = tags*non_padding_mask.squeeze()

    #     # edge_type_tag = tags_full.float().unsqueeze(-1)@tags_full.float().unsqueeze(-2)
    #     # non_self_loop_mask = 1-torch.eye(tags.shape[1],device = edge_type_tag.device)
    #     # edge_type_tag *= non_self_loop_mask

    #     # delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
    #     # dist = delta_pos.norm(dim=-1)
    #     # edge_dirs = delta_pos / (dist.unsqueeze(-1) + 1e-8)
    #     # edge_label = (deltapos_node.unsqueeze(-2)@edge_dirs.permute(0,1,3,2)).squeeze(-2)###around 95% are positive.largely imbalance. Why?

    #     print("appending ",i )
    #     rec_node_label_full.append(deltapos_node[non_padding_mask.squeeze(-1)])
    #     rec_node_label_center.append(deltapos_node[center_mask.squeeze(-1)])
    #     # rec_edge_type_tags.append(edge_type_tag.view(-1))
    #     # rec_edge_labels.append(edge_label[edge_type_tag>0].view(-1))
        

    # total_node = torch.concat(rec_node_label_full)
    # node_mean = total_node.mean(axis=0)
    # node_std = total_node.std(axis=0)
    # total_node_center = torch.concat(rec_node_label_center)
    # center_node_mean = total_node_center.mean(axis=0)
    # center_node_std = total_node_center.std(axis=0)
    # print("full",node_mean, node_std)
    # print("center",center_node_mean, center_node_std)
    # # edge_mean = torch.concat(rec_edge_labels).mean()
    # # edge_std = torch.concat(rec_edge_labels).std()
    # # edge_type_tags_full = torch.concat(rec_edge_type_tags)
    # # edge_tags_bincount = edge_type_tags_full.detach().cpu().int().view(-1).bincount()
    # # edge_tags_ratio = torch.nn.functional.normalize(edge_tags_bincount.float().unsqueeze(0),p=1)
    # # find_error_node = (((total_node-node_mean)/node_std).max(dim=1)[0]>2.2).sum()

    # target_rec=[]
    # deltapos_rec=[]
    # for i,samples in enumerate(progress):
    #     target_rec.append(samples[0]['targets']['relaxed_energy'])
    #     deltapos_rec.append(samples[0]['targets']['deltapos'])
    #     if i%100==0:
    #         print(i)
    # torch.stack(target_rec).mean() #pcq tensor(5.6823)
    # torch.stack(target_rec).std()   #pcq tensor(1.1575)
    # stackedpos = torch.cat([deltapos.reshape(-1,3) for deltapos in deltapos_rec],dim = -2) ##PCQ tensor([0.4146, 0.5808, 0.6160])
    
    for i, samples in enumerate(progress):
        # if i<300:
        #     if i%10==0:
        #         print(i)
        #     continue
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):  
            log_output = trainer.train_step(samples) #loss is summation (over samples) of recovered(* std) MAE

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


def validate_and_save(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    valid_subsets: List[str],
    end_of_epoch: bool,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf

    # Stopping conditions (and an additional one based on validation loss later
    # on)
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"num_updates: {num_updates} >= max_update: {max_update}"
        )

    training_time_hours = trainer.cumulative_training_time() / (60 * 60)
    if (
        cfg.optimization.stop_time_hours > 0
        and training_time_hours > cfg.optimization.stop_time_hours
    ):
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"cumulative_training_time: {training_time_hours} > "
            f"stop_time_hours: {cfg.optimization.stop_time_hours} hour(s)"
        )

    do_save = (
        (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
        or should_stop
        or (
            cfg.checkpoint.save_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.checkpoint.save_interval_updates == 0
            and num_updates >= cfg.dataset.validate_after_updates
        )
    )
    do_validate = (
        (
            (not end_of_epoch and do_save)  # validate during mid-epoch saves
            or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
            or should_stop
            or (
                cfg.dataset.validate_interval_updates > 0
                and num_updates > 0
                and num_updates % cfg.dataset.validate_interval_updates == 0
            )
        )
        and not cfg.dataset.disable_validation
        and num_updates >= cfg.dataset.validate_after_updates
    )

    # Validate
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)

    should_stop |= should_stop_early(cfg, valid_losses[0])

    # Save checkpoint
    if do_save or should_stop:
        checkpoint_utils.save_checkpoint(
            cfg.checkpoint, trainer, epoch_itr, valid_losses[0]
        )

    return valid_losses, should_stop


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    subsets: List[str],
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    mix_reg_cls = cfg.model.mix_reg_cls
    for subset in subsets:
        if cfg.model.ensembled_testing!=-1:
            if mix_reg_cls:
                pred_cls_set_ens, pred_reg_set_ens, label_cls_set_ens, label_reg_set_ens,is_reg_set_ens = [],[],[],[],[]
            else:
                pred_set_ens, label_set_ens = [],[]

            for frame_id in range(cfg.model.ensembled_testing):
                logger.info('begin validation on "{}" subset, frame id is "{}" '.format(subset, frame_id))

                # Initialize data iterator
                itr = trainer.get_valid_iterator(subset).next_epoch_itr(
                    shuffle=False, set_dataset_epoch=False  # use a fixed valid set
                )
                if cfg.common.tpu:
                    itr = utils.tpu_data_loader(itr)
                progress = progress_bar.progress_bar(
                    itr,
                    log_format=cfg.common.log_format,
                    log_interval=cfg.common.log_interval,
                    epoch=epoch_itr.epoch,
                    prefix=f"valid on '{subset}' subset",
                    tensorboard_logdir=(
                        cfg.common.tensorboard_logdir
                        if distributed_utils.is_master(cfg.distributed_training)
                        else None
                    ),
                    default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
                    wandb_project=(
                        cfg.common.wandb_project
                        if distributed_utils.is_master(cfg.distributed_training)
                        else None
                    ),
                    wandb_run_name=os.environ.get(
                        "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
                    ),
                )

                # create a new root metrics aggregator so validation metrics
                # don't pollute other aggregators (e.g., train meters)
                
                if mix_reg_cls:
                    pred_cls_set, pred_reg_set, label_cls_set, label_reg_set,is_reg_set,smi,property_name = [],[],[],[],[],[],[]
                else:
                    pred_set, label_set, smi, property_name = [],[],[],[]

                start_time = time.time()
                with metrics.aggregate(new_root=True) as agg:
                    for i, sample in enumerate(progress):
                        
                        if (
                            cfg.dataset.max_valid_steps is not None
                            and i > cfg.dataset.max_valid_steps
                        ):
                            break
                        sample['net_input']['pos']=sample['net_input']['pos'][:,frame_id,:,:]
                        sample['targets']['deltapos']=sample['targets']['deltapos'][:,frame_id,:,:]
                        if mix_reg_cls:
                            _, pred_cls, pred_reg, label_cls, label_reg, is_reg= trainer.valid_step(sample)
                            pred_cls_set.append(pred_cls)
                            pred_reg_set.append(pred_reg)
                            label_cls_set.append(label_cls)
                            label_reg_set.append(label_reg)
                            is_reg_set.append(is_reg)
                        else:
                            _, pred, label= trainer.valid_step(sample)
                            pred_set.append(pred)
                            label_set.append(label)
                        smi.append(sample['task_input'].get('smi','no_smi')) 
                        property_name.append(sample['task_input'].get('property_name',"no_property_name"))
                end_time = time.time()

                if mix_reg_cls:
                    is_reg_set = torch.cat(is_reg_set,dim=-1).detach().cpu()
                    pred_reg_set = torch.cat(pred_reg_set,dim=-1).detach().cpu()
                    label_reg_set = torch.cat(label_reg_set,dim=-1).detach().cpu()
                    pred_cls_set = [t for t in pred_cls_set if t.nelement()>0]
                    pred_cls_set = torch.cat(pred_cls_set).detach().cpu()
                    label_cls_set = torch.cat(label_cls_set,dim=-1).detach().cpu()
                    # is_reg_set_ens.append(is_reg_set)
                    pred_reg_set_ens.append(pred_reg_set)
                    # label_reg_set_ens.append(label_reg_set)
                    pred_cls_set_ens.append(pred_cls_set)
                    # label_cls_set_ens.append(label_cls_set)
                    instance_num = is_reg_set.shape[0]
                else:
                    pred_set = torch.cat(pred_set,dim=-1).detach().cpu()
                    label_set = torch.cat(label_set,dim=-1).detach().cpu()
                    pred_set_ens.append(pred_set)
                    instance_num = label_set.shape[0]
                    # label_set_ens.append(label_set)


            ##average the prediction of the pred_reg_set_ens and pred_cls_set_ens
            pred_set = torch.stack(pred_reg_set_ens,dim=0).mean(dim=0) 
            pred_cls_set = torch.stack(pred_cls_set_ens,dim=0).mean(dim=0)
            label_set = label_reg_set
        else:
            logger.info('begin validation on "{}" subset'.format(subset))
            # Initialize data iterator
            itr = trainer.get_valid_iterator(subset).next_epoch_itr(
                shuffle=False, set_dataset_epoch=False  # use a fixed valid set
            )
            if cfg.common.tpu:
                itr = utils.tpu_data_loader(itr)
            progress = progress_bar.progress_bar(
                itr,
                log_format=cfg.common.log_format,
                log_interval=cfg.common.log_interval,
                epoch=epoch_itr.epoch,
                prefix=f"valid on '{subset}' subset",
                tensorboard_logdir=(
                    cfg.common.tensorboard_logdir
                    if distributed_utils.is_master(cfg.distributed_training)
                    else None
                ),
                default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
                wandb_project=(
                    cfg.common.wandb_project
                    if distributed_utils.is_master(cfg.distributed_training)
                    else None
                ),
                wandb_run_name=os.environ.get(
                    "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
                ),
            )

            # create a new root metrics aggregator so validation metrics
            # don't pollute other aggregators (e.g., train meters)
            
            if mix_reg_cls:
                pred_cls_set, pred_reg_set, label_cls_set, label_reg_set,is_reg_set,smi, property_name = [],[],[],[],[],[],[]
                moe_attention_full_set,task_idx_set = [],[]
            else:
                pred_set, label_set,smi,property_name = [],[],[],[]
            record_data_path = cfg.checkpoint.save_dir+"/record"
            # if not os.path.exists(record_data_path):
            #     os.mkdir(record_data_path)
            try: 
                os.makedirs(record_data_path, exist_ok = True)
            except:
                pass
                
            start_time = time.time()
            with metrics.aggregate(new_root=True) as agg:
                for i, sample in enumerate(progress):
                    if i%100==0:
                        print("validated",i)
                    if (
                        cfg.dataset.max_valid_steps is not None
                        and i > cfg.dataset.max_valid_steps
                    ):
                        break
                    if mix_reg_cls:
                        if cfg.model.moe_attention_full:
                            _, pred_cls, pred_reg, label_cls, label_reg, is_reg, moe_attentions_full,task_idx= trainer.valid_step(sample)
                        else:
                            _, pred_cls, pred_reg, label_cls, label_reg, is_reg= trainer.valid_step(sample)
                        pred_cls_set.append(pred_cls[~is_reg])
                        pred_reg_set.append(pred_reg[is_reg])
                        label_cls_set.append(label_cls[~is_reg])
                        label_reg_set.append(label_reg[is_reg])
                        is_reg_set.append(is_reg)
                        if cfg.model.moe_attention_full:
                            moe_attention_full_set.append(moe_attentions_full)
                            task_idx_set.append(task_idx)
                    else:
                        _, pred, label= trainer.valid_step(sample)
                        pred_set.append(pred)
                        label_set.append(label)
                    # import pdb
                    # pdb.set_trace()
                    if len(sample.keys())!=0:
                        smi.append(sample['task_input'].get('smi','no_smi')) 
                        property_name.append(sample['task_input'].get('property_name',"no_property_name"))
                    else:
                        smi.append(['padding_smi']*(label_reg.shape[0]+label_cls.shape[0]))
                        property_name.append(['padding_property_name']*(label_reg.shape[0]+label_cls.shape[0]))
            end_time = time.time()
            # moe_attention_full_set_structured = torch.stack([torch.stack([torch.stack(moe_attention_full_set[j][i]) 
            #                                                               for i in range(len(moe_attention_full_set[0]))]) 
            #                                                               for j in range(len(moe_attention_full_set))]) #[instance_num, 2(before, after),layers (12), batch, moe ]
            # task_idx_set_structured = torch.stack(task_idx_set)

            # # moe_attention_full_set_structured = moe_attention_full_set_structured.permute([0,3,1,2,4]).reshape([moe_attention_full_set_structured.shape[0],moe_attention_full_set_structured.shape[-2],-1])
            # moe_attention_full_set_structured = moe_attention_full_set_structured.permute([0,3,1,2,4])[:,:,0,0,:]
            # moe_attention_full_set_cleaned = torch.stack([moe_attention_full_set_structured[task_idx_set_structured==i][0] for i in task_idx_set_structured.unique()])
            # idx_unique = task_idx_set_structured.unique()
            # torch.save(moe_attention_full_set_cleaned,record_data_path+"/task_emb.pt")
            # torch.save(idx_unique,record_data_path+"/task_idx.pt")

            if mix_reg_cls:
                pred_set = torch.cat(pred_reg_set,dim=-1).detach().cpu()
                label_set = torch.cat(label_reg_set,dim=-1).detach().cpu()
                is_reg_set = torch.cat(is_reg_set,dim=-1).detach().cpu()
                instance_num = is_reg_set.shape[0]
                pred_cls_set = [t for t in pred_cls_set if t.nelement()>0]
                if pred_cls_set != []:
                    pred_cls_set = torch.cat(pred_cls_set).detach().cpu()
                else:
                    pred_cls_set = torch.tensor([])
                label_cls_set = torch.cat(label_cls_set,dim=-1).detach().cpu()

            else:
                pred_set = torch.cat(pred_set,dim=-1).detach().cpu()
                label_set = torch.cat(label_set,dim=-1).detach().cpu()
                is_reg_set = torch.ones(label_set.shape[0]).to(torch.bool)
                instance_num = pred_set.shape[0]
                pred_cls_set = torch.zeros(0,2)
                label_cls_set = torch.zeros(0)


        smi_combined, property_name_combined = [],[]
        for sublist in smi:
            smi_combined.extend(sublist)
        for sublist in property_name:
            property_name_combined.extend(sublist)
        # label_set = label_set if isinstance(label_set, np.ndarray) else label_set.numpy()
        # pred_set = pred_set if isinstance(pred_set, np.ndarray) else pred_set.numpy()

        roc_auc_all = float('nan')
        acc_all = float('nan')
        r2_all = float('nan')
        mae_all = float('nan')
        f1 = float('nan')

        ## Calculate Overall Metric
        if mix_reg_cls:
            if label_cls_set.shape[0]:
                roc_auc_all = sklearn.metrics.roc_auc_score(nn.functional.one_hot(label_cls_set.long(), num_classes=2), pred_cls_set)
                acc_all = sklearn.metrics.accuracy_score(label_cls_set, pred_cls_set.argmax(dim = -1))
                f1 = sklearn.metrics.f1_score(label_cls_set, pred_cls_set.argmax(dim = -1))
            else:
                ## Calculate mock classification metric for chiral
                from sklearn.metrics import accuracy_score, f1_score
                # Convert to binary class labels
                # threshold = 1 #Chial Cliff
                threshold = 0 #l/d chiral classification
                pred_binary = (pred_set > threshold).float()
                label_binary = (label_set > threshold).float()

                # Convert to numpy arrays for sklearn metrics
                pred_binary_np = pred_binary.numpy()
                label_binary_np = label_binary.numpy()

                # Calculate accuracy and F1 score
                acc_all = accuracy_score(label_binary_np, pred_binary_np)
                f1 = f1_score(label_binary_np, pred_binary_np)
                ## End of mock metric for chiral

        print("Overall: roc_auc",roc_auc_all,"acc",acc_all, "f1",f1)

        if pred_set.shape[0]!=0:
            _,_,r_value,_,_ = scipy.stats.linregress(pred_set,label_set)
            r2_all = r_value**2
            mae_all = (pred_set-label_set).abs().mean().item()
        print("Overall: r_square",r2_all,"mae",mae_all)

        file_path = os.path.join(record_data_path, 'metric_all.csv')

        # Data to be written or appended
        # import pdb
        # pdb.set_trace()
        # print(sample.keys())
        if len(set(property_name_combined))>1:
            all_name = "all"
        else:
            all_name = property_name_combined[0]
        data = {
            'Property Name': all_name,
            'Steps': trainer.get_num_updates(),
            'Epoch': epoch_itr.epoch,
            'ROC_AUC': roc_auc_all,
            'Accuracy': acc_all,
            'F1 Score': f1,
            'r_square': r2_all,
            'mae':mae_all,
        }
        
        lock = FileLock(file_path+".lock")
        with lock:
            # Check if file exists
            if not os.path.exists(file_path):
                # Create a new DataFrame and save it as CSV
                df = pd.DataFrame([data])
                df.to_csv(file_path, index=False)
            else:
                try:
                    # Read existing file, append new data, and save it
                    df = pd.read_csv(file_path)
                except pd.errors.EmptyDataError:
                    df = pd.DataFrame(columns=data.keys())
                df = df.append(data, ignore_index=True)
                df.to_csv(file_path, index=False)

        print("1000 molecule inference time:",round((end_time-start_time)/instance_num*1000,2) , "s") 

        ### Detailed groups   
        if label_cls_set.shape[0]:
            data_cls = {
                'property_name': [p for p, flag in zip(property_name_combined, ~is_reg_set.numpy()) if flag],
                'smi': [p for p, flag in zip(smi_combined, ~is_reg_set.numpy()) if flag],
                'label': label_cls_set.numpy(),
                'prediction': pred_cls_set.numpy().argmax(axis=-1),  # Only keep the predicted class
            }
            # Adding columns for raw predictions
            for i in range(pred_cls_set.shape[1]):
                data_cls[f'column_{i}'] = pred_cls_set[:, i]

            df_cls = pd.DataFrame(data_cls)
            grouped_cls = df_cls.groupby('property_name')
            acc_group = []
            roc_auc_group = []
            name_group = []
            for name, group in grouped_cls:
                # Save group to CSV
                # group.to_csv(f'{name}.csv', index=False)
                
                # Calculate metrics
                labels = group['label'].values
                preds = group[[f'column_{i}' for i in range(pred_cls_set.shape[1])]].values
                roc_auc,acc = 0,0
                if len(preds) > 1:
                    try:
                        roc_auc = sklearn.metrics.roc_auc_score(nn.functional.one_hot(torch.tensor(labels).long(), num_classes=2), preds)
                    except ValueError:
                        pass
                    acc = sklearn.metrics.accuracy_score(labels, group['prediction'].values)
                    print(f"For property {name}:","Acc:",acc,"Roc_auc", roc_auc)
                    acc_group.append(acc)
                    roc_auc_group.append(roc_auc)
                    name_group.append(name)
                else:
                    print(f"For property {name}: Not enough data points for classification metrics")
            acc_group = np.array(acc_group)
            roc_auc_group = np.array(roc_auc_group)
            print(f"Task_Average:","Acc:",acc_group.mean(),"Roc_auc", roc_auc_group.mean())
            name_group = np.append(name_group,"Task Average")
            acc_group = np.append(acc_group,acc_group.mean())
            roc_auc_group = np.append(roc_auc_group,roc_auc_group.mean())
            cls_task_df = pd.DataFrame({'property_name':name_group,'acc':acc_group,'roc_auc':roc_auc_group})
            cls_task_df.to_csv(record_data_path+"/cls_task_step"+str(trainer.get_num_updates())+"_epoch"+str(epoch_itr.epoch)+".csv",index=False)

                
        data_reg = {
            'property_name': [p for p, flag in zip(property_name_combined, is_reg_set.numpy()) if flag],
            'smi': [p for p, flag in zip(smi_combined, is_reg_set.numpy()) if flag],
            'label': label_set,
            'prediction': pred_set,
        }

        # Create a DataFrame from the data

        df_reg = pd.DataFrame(data_reg)
        
        # Group by property_name

        grouped_reg = df_reg.groupby('property_name')

        # Process each group

        mae_group = []
        r2_group = []
        name_group = []
        for name, group in grouped_reg:
            # Save group to CSV
            # group.to_csv(f'{name}_reg.csv', index=False)
            
            # Calculate metrics
            labels = group['label'].values
            preds = group['prediction'].values
            
            if len(preds) > 1:
                _, _, r_value, _, _ = scipy.stats.linregress(preds, labels)
                r2 = r_value**2
                mae = np.abs(preds - labels).mean()
                print(f"For property {name}: R2 = {r2}, MAE = {mae}")
                r2_group.append(r2)
                mae_group.append(mae)
                name_group.append(name)
            else:
                print(f"For property {name}: Not enough data points for regression metrics")
        r2_group = np.array(r2_group)
        mae_group = np.array(mae_group)
        print(f"Task_Average:","r2", r2_group.mean(),"mae:",mae_group.mean())
        name_group.append("Task Average")
        r2_group = np.append(r2_group,r2_group.mean())
        mae_group = np.append(mae_group,mae_group.mean())
        reg_task_df = pd.DataFrame({'property_name':name_group,'r2':r2_group,'mae':mae_group})
        reg_task_df.to_csv(record_data_path+"/reg_task_step"+str(trainer.get_num_updates())+"_epoch"+str(epoch_itr.epoch)+".csv",index=False)

        if cfg.model.save_as_csv:
            # To create a single CSV file named according to unique property names
            unique_properties = sorted(list(set([p for p, flag in zip(property_name_combined, ~is_reg_set.numpy()) if flag])))
            filename = "./results_csv/cls_"+ "_".join(unique_properties) + ".csv"
            # if filename is too long, then change to a shorter one according to number of unique properties.
            if len(filename)>255:
                filename = "./results_csv/cls_"+ str(len(unique_properties)) + "properties.csv"
            df_cls.to_csv(filename, index=False)
            # To create a single CSV file named according to unique property names
            unique_properties = sorted(list(set([p for p, flag in zip(property_name_combined, is_reg_set.numpy()) if flag])))
            filename = "./results_csv/reg_" + "_".join(unique_properties) + ".csv"
            # if filename is too long, then change to a shorter one according to number of unique properties.
            if len(filename)>255:
                filename = "./results_csv/reg_"+ str(len(unique_properties)) + "properties.csv"
            df_reg.to_csv(filename, index=False)

        stats = agg.get_smoothed_values()
        # log validation stats

        # plt.savefig("./update"+str(stats['num_updates']))
        # plt.close()

        stats['R2'] = r2_all
        stats['mae'] = mae_all
        if mix_reg_cls:
            stats['roc_auc'] = roc_auc_all
            stats['acc'] = acc_all
            stats['f1'] = f1
            stats['R2_acc_mean'] = (r2_all+acc_all)/2
        stats = get_valid_stats(cfg, trainer, stats)
        if hasattr(task, "post_validate"):
            task.post_validate(trainer.get_model(), stats, agg)

        progress.print(stats, tag=subset, step=trainer.get_num_updates())   
        valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])

    return valid_losses

def test_and_save(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    valid_subsets: List[str],
    end_of_epoch: bool,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf

    # Stopping conditions (and an additional one based on validation loss later
    # on)
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"num_updates: {num_updates} >= max_update: {max_update}"
        )

    training_time_hours = trainer.cumulative_training_time() / (60 * 60)
    if (
        cfg.optimization.stop_time_hours > 0
        and training_time_hours > cfg.optimization.stop_time_hours
    ):
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"cumulative_training_time: {training_time_hours} > "
            f"stop_time_hours: {cfg.optimization.stop_time_hours} hour(s)"
        )

    do_save = (
        (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
        or should_stop
        or (
            cfg.checkpoint.save_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.checkpoint.save_interval_updates == 0
            and num_updates >= cfg.dataset.validate_after_updates
        )
    )
    do_validate = (
        (
            (not end_of_epoch and do_save)  # validate during mid-epoch saves
            or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
            or should_stop
            or (
                cfg.dataset.validate_interval_updates > 0
                and num_updates > 0
                and num_updates % cfg.dataset.validate_interval_updates == 0
            )
        )
        and not cfg.dataset.disable_validation
        and num_updates >= cfg.dataset.validate_after_updates
    )

    # Validate
    valid_losses = [None]
    testit(cfg, trainer, task, epoch_itr, valid_subsets)

    should_stop |= should_stop_early(cfg, valid_losses[0])

    # Save checkpoint
    if do_save or should_stop:
        checkpoint_utils.save_checkpoint(
            cfg.checkpoint, trainer, epoch_itr, valid_losses[0]
        )

    return valid_losses, should_stop


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats

def testit(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    subsets: List[str],
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    # valid_losses = []
    predictions={}
    for subset in subsets:
        logger.info('begin testing on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(
            shuffle=False, set_dataset_epoch=False  # use a fixed valid set
        )
        if cfg.common.tpu:
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"test on '{subset}' subset",
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            wandb_run_name=os.environ.get(
                "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
            ),
        )

        predictions[subset[5:]+"_ids"]=[]
        predictions[subset[5:]+"_energy"]=[]
        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for i, sample in enumerate(progress):
                if (
                    cfg.dataset.max_valid_steps is not None
                    and i > cfg.dataset.max_valid_steps
                ):
                    break
                sid, energy = trainer.test_step(sample)
                predictions[subset[5:]+"_ids"].extend([str(k) for k in sid.tolist()])
                predictions[subset[5:]+"_energy"].extend(energy.tolist())
                if i%100==0:
                    logger.info('inference i "{}" '.format(i))
                # if i>10: break

    results_file_path = os.path.join(cfg.checkpoint.save_dir,cfg.checkpoint.restore_file.split(".")[0]+"_submit.npz")
    np.savez_compressed(
        results_file_path,
        **predictions
    )
    logger.info('testing result saved in "{}" '.format(results_file_path))
    # exit()
        # log validation stats
        # stats = get_valid_stats(cfg, trainer, agg.get_smoothed_values())

        # if hasattr(task, "post_validate"):
        #     task.post_validate(trainer.get_model(), stats, agg)

        # progress.print(stats, tag=subset, step=trainer.get_num_updates())

        # valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
    # return valid_losses
    return



def get_valid_stats(
    cfg: DictConfig, trainer: Trainer, stats: Dict[str, Any]
) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[cfg.checkpoint.best_checkpoint_metric],
        )
    return stats


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)
    if args.sweep_mode:
        wandb.init(project=args.wandb_project, reinit=False)
        
        for key in wandb.config.keys():
            if hasattr(args, key):
                if isinstance(getattr(args,key),list):
                    setattr(args, key, [wandb.config[key]])
                else:
                    setattr(args, key, wandb.config[key])
    cfg = convert_namespace_to_omegaconf(args)

    if cfg.common.use_plasma_view:
        server = PlasmaStore(path=cfg.common.plasma_path)
        logger.info(
            f"Started plasma server pid {server.server.pid} {cfg.common.plasma_path}"
        )
    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)

    # if cfg.common.use_plasma_view:
    #     server.server.kill()


if __name__ == "__main__":
    cli_main()
