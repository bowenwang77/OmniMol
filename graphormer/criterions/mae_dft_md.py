# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Mapping, Sequence, Tuple
from numpy import mod
import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn

from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from ..modules.visualize import visualize_trace, visualize_trace_poscar, plot_mol_with_attn, visualize_trace_poscar_molecule

@register_criterion("mae_dft_md")
class MAE_DFT_MD(FairseqCriterion):
    e_thresh = 0.2
    e_mean = 0
    e_std = 1
    #node norm unclean center
    # d_mean = [0.1353900283575058, 0.06877671927213669, 0.08111362904310226]
    # d_std = [1.7862379550933838, 1.78688645362854, 0.8023099899291992]
    #node norm cleaned center
    d_mean = [0.0,0.0,0.0]
    d_std = [1, 1, 1]    
    # print("node norm changed")
    def __init__(self, task, cfg):
        super().__init__(task)
        self.node_loss_weight = cfg.node_loss_weight
        MAE_DFT_MD.distributed_world_size = cfg.distributed_world_size
        MAE_DFT_MD.update_freq = cfg.update_freq[0]
        if cfg.use_shift_proj:
            self.edge_loss_weight = cfg.edge_loss_weight
            self.min_edge_loss_weight = cfg.min_edge_loss_weight
            self.edge_loss_weight_range = max(
                0, self.edge_loss_weight - self.min_edge_loss_weight
            )

        self.min_node_loss_weight = cfg.min_node_loss_weight
        self.max_update = cfg.max_update
        self.node_loss_weight_range = max(
            0, self.node_loss_weight - self.min_node_loss_weight
        )

        self.jac_loss_weight = cfg.jac_loss_weight
        self.compute_jac_loss = cfg.compute_jac_loss
        self.use_shift_proj = cfg.use_shift_proj
        self.no_node_mask = cfg.no_node_mask
        self.get_inter_pos_trace = cfg.get_inter_pos_trace
        self.visualize = cfg.visualize
        self.noisy_node_weight = cfg.noisy_node_weight
        self.explicit_pos = cfg.explicit_pos
        self.l2_node_loss = cfg.l2_node_loss
        self.fix_atoms = cfg.fix_atoms
        self.noise_scale = cfg.noise_scale
        self.max_cls_num = cfg.max_cls_num
        self.cls_weight = cfg.cls_weight
        self.mix_reg_cls = cfg.mix_reg_cls
        self.readout_attention = cfg.readout_attention
        self.use_meta = cfg.use_meta
        self.distributed_world_size = cfg.distributed_world_size
        self.save_dir = cfg.save_dir
        # if cfg.use_unnormed_node_label:
        #     MAE_DFT_MD.d_mean = [0.0,0.0,0.0]
        #     MAE_DFT_MD.d_std = [self.noise_scale,self.noise_scale,self.noise_scale]
        # print("node norm", MAE_DFT_MD.d_mean,MAE_DFT_MD.d_std)

    def weighted_binary_cross_entropy(self,output, target, weights=None, keep_dim=False):
        if weights is not None:
            assert len(weights.shape) == len(target.shape)
            output = output.float()
            target = target.float()
            loss = -1* target * torch.log(output+1e-8)
            loss = loss * weights
            if keep_dim:
                return loss
            else:
                return loss.sum(dim=1).mean()
        else:
            return torch.nn.functional.binary_cross_entropy(output, target)

    def forward(
        self,
        model: Callable[..., Tuple[Tensor, Tensor, Tensor]],
        sample: Mapping[str, Mapping[str, Tensor]],
        reduce=True,
    ):
        if 'targets' in sample.keys():

            update_num = model.num_updates
            assert update_num >= 0
            node_loss_weight = (
                self.node_loss_weight
                - self.node_loss_weight_range * update_num / self.max_update
            )
            if self.use_shift_proj:
                edge_loss_weight = (
                    self.edge_loss_weight
                    - self.edge_loss_weight_range * update_num / self.max_update
                )

            use_noisy_node = False
            # import pdb
            # pdb.set_trace()
            sid =sample["task_input"]['sid']
            cell = sample["task_input"]['cell']
            smi = sample["task_input"]['smi']
            prop_name = sample["task_input"]['property_name']
            task_mean = sample["task_input"]["task_mean"]+1e-8
            task_std = sample["task_input"]["task_std"]
            prop_id = sample["task_input"]["prop_id"]
            task_geo_mean = sample["task_input"]["task_geo_mean"]
            task_geo_std = sample["task_input"]["task_geo_std"]

            sample["net_input"]['step']=update_num

            non_padding_mask = sample["net_input"]["atoms"].ne(0) ##also removes nodes that passes boundary. correlated codes in is2re.py
            valid_nodes = non_padding_mask.sum()
            task_idx = sample["net_input"]["task_idx"]
            is_reg = sample["net_input"]["is_reg"]
            if self.use_meta:
                meta = {'task_mean':task_mean, 'task_std':task_std, 'prop_id':prop_id}
                sample["net_input"]["meta"] = meta
            

            output_vals = model(**sample["net_input"],)   

            # output = output_vals['eng_output']
            ##Output include regression and classification, therefore we need to mask the output by "is_reg"
            ##And divide the output into output_reg and output_cls
            output_reg = output_vals['reg_output']
            output_cls = output_vals['cls_output']
            node_output = output_vals['node_output']
            moe_attentions_full = output_vals['moe_attentions_full']
            visualize_dir = self.save_dir+"/"
            # print(visualize_dir)
            # visualize_dir = visualize+"/"
            # if update_num <100:
            if self.readout_attention and self.visualize and model.training:
            # if sum([name in prop_name for name in ['H-HT']])>0 and sum(tgt_idx in sid for tgt_idx in [1440])>0:
                print("screened")
                output_reg_attn = output_vals['reg_attention']
                output_cls_attn = output_vals['cls_attention']
                node_attention_list=[]
                for i in range(len(node_output)):
                    if is_reg[i]:
                        node_attention_list.append(output_reg_attn[i][non_padding_mask[i]])
                    else:
                        node_attention_list.append(output_cls_attn[i][non_padding_mask[i]])
                for i in range(len(node_attention_list)):
                    # if (sid[i] in [2123,695,137,9676,1440,8265]) and (prop_name[i] in ['Ames','EC', 'F(20%)', 'hERG', 'H-HT', 'LogP']):
                    # if (sid[i] in [1440]) and (prop_name[i] in ['H-HT']):
                    if True:
                        print('Found ', prop_name[i], sid[i])
                        # plot_mol_with_attn(sample["net_input"]["atoms"][i][non_padding_mask[i]].cpu().detach().numpy(), 
                        #                 smi[i], node_attention_list[i].cpu().detach().numpy(),
                        #                 visualize_dir + "visualize/", 
                        #                 idx=sid[i].cpu().detach().numpy(), with_hydrogen = True, 
                        #                 prop_name = prop_name[i], 
                        #                 pos = sample["net_input"]["pos"][i][non_padding_mask[i]].cpu().detach().numpy())
                        plot_mol_with_attn(sample["net_input"]["atoms"][i][non_padding_mask[i]].cpu().detach().numpy(), 
                                        smi[i], node_attention_list[i].cpu().detach().numpy(),
                                        visualize_dir + "visualize/", 
                                        idx=sid[i].cpu().detach().numpy(), with_hydrogen = False, 
                                        prop_name = prop_name[i], 
                                        pos = sample["net_input"]["pos"][i][non_padding_mask[i]].cpu().detach().numpy())


            node_target_mask = output_vals['node_target_mask'] #Mask of nodes: in center cell & surface & adsorbate
            edge_dirs = output_vals['edge_dirs']
            # if self.get_inter_pos_trace and not model.training:
            if False:
                deltapos_trace = output_vals['deltapos_trace'] #[number of intermediate trace, batchsize*2, num of atom, 3]
                temp_visualize = True
                temp_visualize_sid = 2044931 #1423867 2044931
                if (sid == temp_visualize_sid).any():
                    temp_visualize=True
                if self.visualize or temp_visualize:
                    # visualize_trace_poscar(sample,cell, sid, output_vals, self.e_mean, self.e_std, self.d_mean, self.d_std, directory="/HOME/DRFormer_ADMET/graphormer/admet_paper/visulization/trace")
                    visualize_trace_poscar_molecule(sample,cell, sid, output_vals, self.e_mean, self.e_std, self.d_mean, self.d_std, directory=visualize_dir+"trace")

            if self.compute_jac_loss:
                jac_loss = output_vals['jac_loss']
                f_deq_nstep = output_vals['f_deq_nstep']
                f_deq_residual = output_vals['f_deq_residual']
            if self.use_shift_proj:
                edge_output = output_vals['fit_scale_pred']
                edge_target_mask = output_vals['edge_target_mask']
            self.drop_edge_training=False
            if 'drop_edge_mask' in output_vals.keys():
                self.drop_edge_training=True
                drop_edge_mask = output_vals['drop_edge_mask']
            graph_label = sample["targets"]["relaxed_energy"]


            sample_size = graph_label.numel()
            reg_sample_size = is_reg.sum()+1e-8
            cls_sample_size = (~is_reg).sum()+1e-8
            reg_graph_label = graph_label
            cls_graph_label = graph_label.masked_fill(is_reg,0)

            reg_graph_label = reg_graph_label.float()
            # relaxed_energy = (relaxed_energy - self.e_mean) / self.e_std
            # # remove outlier based on 3-sigma principle
            # output_reg[output_reg>3]=0
            # output_reg[output_reg<-3]=0

            loss_reg = F.l1_loss(output_reg.float().view(-1), (reg_graph_label-task_mean)/task_std, reduction="none")
            with torch.no_grad():
                energy_within_threshold = (loss_reg.detach() * task_std < self.e_thresh).masked_fill(~is_reg,0).sum()
            if is_reg.sum()==0:
                loss_reg = torch.tensor(0.0).to(output_reg.device)
            else:
                loss_reg = loss_reg.masked_fill(~is_reg,0).sum()

            if (~is_reg).sum()==0:
                loss_cls = torch.tensor(0.0).to(output_cls.device)
            else:
                one_hot_cls_label = nn.functional.one_hot(cls_graph_label.to(torch.int64), num_classes=self.max_cls_num).float()
                ## Each instance a batch may belong to a different tasks, with highly imbalanced class distribution. T
                # herefore we need to pass additional "cls_weight" to the loss function class weights to calculate the loss,
                #  rather than directly use cross entropy loss. The class_weight shaped as [batch_size, class_number] is calculated as 1/(2*class_mean) for each class.
                #  The output of the model "output_cls" is a [batch_size, class_number] tensor, which is the probability of each class.
                #  The label of the class "one_hot_cls_label" is a [batch_size, class_number] tensor with 0 or 1. The loss is calculated as the cross entropy loss 
                # between the output and the label, weighted by the class_weight. 
                
                cls_loss_weight = torch.stack([1/(2*(1-task_mean+1e-8)),1/(2*(task_mean-1e-8))]).T
                loss_cls = self.weighted_binary_cross_entropy(output_cls.masked_fill(is_reg.unsqueeze(-1),0.5), one_hot_cls_label, cls_loss_weight, keep_dim = True)
                loss_cls = loss_cls.masked_fill(is_reg.unsqueeze(-1),0)
                loss_cls = loss_cls.sum(dim=-1).sum(dim=-1)

                # loss_cls = F.cross_entropy(output_cls, one_hot_cls_label, reduction="none",)
                # loss_cls = loss_cls.sum()

            accuracy = (output_cls.argmax(dim=-1) == cls_graph_label)[~is_reg].sum()
            
            loss_graph = self.cls_weight*loss_cls/cls_sample_size+loss_reg/reg_sample_size
            # print("loss_reg",loss_reg,reg_sample_size)


            deltapos = sample["targets"]["deltapos"].float()
            # deltapos = (deltapos - deltapos.new_tensor(self.d_mean)) / deltapos.new_tensor(
            #     self.d_std
            # )
            deltapos = (deltapos - task_geo_mean.unsqueeze(1)) / task_geo_std.unsqueeze(1)

            ##Consider supervision of nodes: subsurface+surface+adsorbate:
            # if self.no_node_mask:
                # node_loss_center = F.l1_loss(node_output.float()[node_target_mask.squeeze(-1)],deltapos[node_target_mask.squeeze(-1)])
                # node_loss_boundary = F.l1_loss(node_output.float()[~node_target_mask.squeeze(-1)],deltapos[~node_target_mask.squeeze(-1)])
            deltapos_center = deltapos * node_target_mask
            node_output_center = node_output * node_target_mask
            target_cnt_center = node_target_mask.sum(dim=[1, 2])+1e-8
            if self.l2_node_loss:
                node_loss_center = (node_output_center.float()-deltapos_center).norm(dim=-1).sum(dim=-1)/target_cnt_center
            else:
                node_loss_center = (
                    F.l1_loss(node_output_center.float(), deltapos_center, reduction="none")
                    .mean(dim=-1)
                    .sum(dim=-1)
                    / target_cnt_center
                )
            node_loss = node_loss_center.mean()

            if self.use_shift_proj:
                # check number of edge types: edge_target_mask.detach().cpu().int().view(-1).bincount()
                if self.explicit_pos:
                    edge_label = ((sample["targets"]["deltapos"]-output_vals['final_delta_pos']).float().unsqueeze(-2)@edge_dirs.permute(0,1,3,2)) #Raw edge label
                else:
                    edge_label = (sample["targets"]["deltapos"].float().unsqueeze(-2)@edge_dirs.float().permute(0,1,3,2)) #Raw edge label
                edge_label = edge_label.squeeze(-2)


                center_edge_mask = torch.zeros_like(edge_label)
                center_edge_mask = torch.logical_and(center_edge_mask.masked_fill(node_target_mask, 1).bool(), edge_target_mask!=0)
                if self.drop_edge_training:
                    center_edge_mask = torch.logical_and(center_edge_mask, ~drop_edge_mask)
                edge_label_center = edge_label*center_edge_mask
                edge_output_center = edge_output*center_edge_mask
                target_cnt_center = center_edge_mask.sum(dim=[1, 2])

                edge_loss_center = F.l1_loss(edge_output_center, edge_label_center,reduction = "none").sum(dim=-1).sum(dim=-1)/target_cnt_center
                edge_loss = edge_loss_center.mean()
            mix_reg_cls = self.mix_reg_cls

            total_loss = loss_graph + node_loss_weight * node_loss

            logging_output = {
                "loss_graph": loss_graph.detach(),
                "loss_reg": loss_reg.detach(),
                "loss_cls": loss_cls.detach(),
                "accuracy": accuracy.detach(),
                "energy_within_threshold": energy_within_threshold,
                "node_loss": node_loss.detach(),
                "sample_size": sample_size,
                "reg_sample_size": reg_sample_size,
                "cls_sample_size": cls_sample_size,
                "nsentences": sample_size,
                "num_nodes": valid_nodes.detach(),
                "node_loss_weight": node_loss_weight * sample_size,
                "prediction_cls": output_cls.detach().float(),
                "prediction_reg": output_reg.detach().float().view(-1)*task_std+task_mean,
                "label_cls": cls_graph_label.detach(),
                "label_reg": reg_graph_label.detach(),
                "mix_reg_cls": mix_reg_cls,
                "is_reg": is_reg.detach(),
                "task_idx": task_idx,
                # "moe_attentions_full": moe_attentions_full,
            }
            if self.distributed_world_size == 1:
                logging_output["moe_attentions_full"] = moe_attentions_full

            if self.compute_jac_loss:
                logging_output["jac_loss"] = jac_loss.detach()
                logging_output["f_deq_nstep"] = f_deq_nstep
                logging_output["f_deq_residual"] = f_deq_residual
                total_loss = total_loss + self.jac_loss_weight * jac_loss
            
            if self.use_shift_proj:
                logging_output["edge_loss_weight"]=edge_loss_weight * sample_size
                logging_output["edge_loss"] = edge_loss.detach()
                total_loss = total_loss + edge_loss_weight * edge_loss

            logging_output["loss"] = total_loss.detach()
            return total_loss, sample_size, logging_output
        else:
            update_num = -1
            sample_size = sample['net_input']['pos'].shape[0]
            sample["net_input"]['step']=update_num
            non_padding_mask = sample["net_input"]["atoms"].ne(0) ##also removes nodes that passes boundary. correlated codes in is2re.py
            valid_nodes = non_padding_mask.sum()
            sid = sample['net_input'].pop('sid')
            cell = sample["net_input"].pop('cell')

            output_vals = model(**sample["net_input"],)   
            energy = output_vals['eng_output']*self.e_std+self.e_mean
            return sid, energy, sample_size, None
    @staticmethod
    def reduce_metrics(logging_outputs: Sequence[Mapping]) -> None:
        loss_total_sum = sum(log.get("loss", 0) for log in logging_outputs)
        # print("loss_total_sum",loss_total_sum)
        loss_graph_sum = sum(log.get("loss_graph", 0) for log in logging_outputs)
        energy_within_threshold_sum = sum(
            log.get("energy_within_threshold", 0) for log in logging_outputs
        )
        node_loss_sum = sum(log.get("node_loss", 0) for log in logging_outputs)
        edge_loss_sum = sum(log.get("edge_loss",0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        jac_loss_sum = sum(log.get("jac_loss",0) for log in logging_outputs)
        reg_loss_sum = sum(log.get("loss_reg", 0) for log in logging_outputs)
        cls_loss_sum = sum(log.get("loss_cls", 0) for log in logging_outputs)
        reg_sample_size = sum(log.get("reg_sample_size", 0) for log in logging_outputs)
        cls_sample_size = sum(log.get("cls_sample_size", 0) for log in logging_outputs)
        accuracy_sum = sum(log.get("accuracy", 0) for log in logging_outputs)
        f_deq_nstep_sum = sum(log.get("f_deq_nstep",0) for log in logging_outputs)
        f_deq_residual_sum = sum(log.get("f_deq_residual",0) for log in logging_outputs)
        ##If the value is using "mean" for each batch, then we should divide by "total_batch_num" instead of "sample_size"

        total_batch_num = len(logging_outputs)*MAE_DFT_MD.distributed_world_size*MAE_DFT_MD.update_freq
        mean_jac_loss = jac_loss_sum /total_batch_num
        mean_f_deq_nstep = f_deq_nstep_sum/ total_batch_num
        mean_f_deq_residual = f_deq_residual_sum / total_batch_num
        mean_loss_graph = loss_graph_sum / total_batch_num
        mean_loss_total = loss_total_sum / total_batch_num
        mean_reg_loss = (reg_loss_sum /reg_sample_size ) * MAE_DFT_MD.e_std
        mean_cls_loss = cls_loss_sum /cls_sample_size
        mean_accuracy = accuracy_sum / cls_sample_size

        energy_within_threshold = energy_within_threshold_sum / total_batch_num
        mean_node_loss = (node_loss_sum / total_batch_num) * sum(MAE_DFT_MD.d_std) / 3.0
        mean_edge_loss = (edge_loss_sum / total_batch_num)
        mean_n_nodes = (
            sum([log.get("num_nodes", 0) for log in logging_outputs]) / sample_size
        )
        node_loss_weight = (
            sum([log.get("node_loss_weight", 0) for log in logging_outputs])
            / sample_size
        )
        edge_loss_weight = (
            sum([log.get("edge_loss_weight", 0) for log in logging_outputs])
            / sample_size
        )
        metrics.log_scalar("loss", mean_loss_total, sample_size, round=6) 
        metrics.log_scalar("loss_graph", mean_loss_graph, sample_size, round=6) 
        metrics.log_scalar("loss_reg", mean_reg_loss, reg_sample_size, round=6)
        metrics.log_scalar("loss_cls", mean_cls_loss, cls_sample_size, round=6)
        metrics.log_scalar("mean_accuracy", mean_accuracy, cls_sample_size, round=6)
        metrics.log_scalar("ewth", energy_within_threshold, sample_size, round=6)
        metrics.log_scalar("node_loss", mean_node_loss, sample_size, round=6)
        metrics.log_scalar("edge_loss", mean_edge_loss, sample_size, round=6)
        metrics.log_scalar("nodes_per_graph", mean_n_nodes, sample_size, round=6)
        metrics.log_scalar("node_loss_weight", node_loss_weight, sample_size, round=6)
        metrics.log_scalar("edge_loss_weight", edge_loss_weight, sample_size, round=6)
        metrics.log_scalar("jac_loss", mean_jac_loss, sample_size, round = 6)
        metrics.log_scalar("f_deq_nstep", mean_f_deq_nstep, sample_size, round = 6)
        metrics.log_scalar("f_deq_residual", mean_f_deq_residual, sample_size, round = 6)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
