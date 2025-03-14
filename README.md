# README for OmniMol

## Introduction

This document outlines the implementation process for OmniMol, a cutting-edge platform designed for molecule property prediction. Please note that due to confidentiality agreements with our data providers, we are unable to release the full proprietary ADMET 2.0 dataset utilized in our research. However, we are actively working on integrating OmniMol as the underlying model for an online platform that will enable users to access and make ADMET predictions and other results presented in this study. We are committed to delivering this platform and will provide updates on its development and release timeline.

For researchers seeking to validate our methodology or conduct similar studies, alternative publicly available datasets can be employed. We recommend using the dataset provided by HelixADMET. Detailed information and access to their dataset can be found at [HelixADMET](https://academic.oup.com/bioinformatics/article/38/13/3444/6590643). Please follow our following instructions for dataset processing.

Additionally, our codebase is developed on the Graphormer framework, which is specifically designed for molecular modeling. For foundational code and setup instructions, please consult the [Graphormer GitHub repository](https://github.com/microsoft/Graphormer).

## Environment Setup

### Conda Environment
Set up the conda environment by following the detailed instructions provided on the Graphormer GitHub page. After setting up the environment, you will need to make the following replacements in the environment directory (suppose you have set up the environment for OmniMol under the directory of `~/research/env/OmniMol`):

- Replace `~/research/env/OmniMol/lib/python3.9/site-packages/fairseq_cli` with `fairseq_mods/fairseq_cli`.
- Replace `~/research/env/OmniMol/lib/python3.9/site-packages/fairseq` with `fairseq_mods/fairseq`.

### Compressed Conda Environment
Alternatively, a more efficient way is to download a pre-configured conda environment using the link below (password: OmniMol2024):

[Download Environment](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155156871_link_cuhk_edu_hk/Eefv8KiXtZpCjInMgzSTzMUBPzj4bTv9Uaedhsnx7bfLCg?e=bvZF23)

After downloading, use the following commands to set up the environment:
```bash
mkdir -p my_env
tar -xzvf [directory/to/]DRLabel_ADMET_20240131.tar -C my_env
source my_env/bin/activate
```

It normally takes no more than 30 minutes for environment setting up.

This environment setup has been tested and is successful on NVIDIA GPUs like TitanX, A40, A100, V100. For NVIDIA 4090, follow the error messages to update the pyg packages as needed.

### Conda environment result
To ensure reproducibility and facilitate the setup process, we have provided a detailed list of the packages used in our conda environment, along with their respective versions and build information. This information can be found in the `OmniMol_env.txt` file, which serves as a reference for users who wish to recreate the exact environment used in our experiments.

## Data Preparation

### CSV Data
For datasets consisting of SMILES strings, place your CSV files under `./example`. The CSV files must contain three columns: SMILES string, group (training, validation, or test), and targeted property. Refer to the provided example of `prop1` and `prop2`.

Run the following commands to process your data:
```bash
python scripts/data_proc_SMILES2lmdb.py
python scripts/task_dic_gene.py
```

Ensure that the `meta_dict.json` file is generated as it is crucial for further steps.

### LMDB Data
Ensure your molecule-property data is in LMDB format. You can reference the structure by inspecting the provided examples in the `example/lmdb` directory.

### Metadata Dictionary
When working with tasks that belong to a specific supergroup, such as "Metabolism" in ADMET prediction for the "CYP2C19-inh" task, it is essential to include these additional details in the property definition. This metadata provides context and facilitates the organization and interpretation of the results.

```json
{
    "CYP2C19-inh": {
        "regression": false,
        "task_idx": 8,
        "cls_num": 2,
        "task_mean": 0.4550738916256158,
        "task_std": 1,
        "prop":"Metabolism",
		"prop_id":3
    },
    "prop1": {
        "regression": false,
        "task_idx": 1,
        "cls_num": 2,
        "task_mean": 0.125,
        "task_std": 1.0
    },
    "prop2": {
        "regression": true,
        "task_idx": 2,
        "cls_num": 1,
        "task_mean": 0.3,
        "task_std": 0.3964124835860459
    }
}
```

To incorporate the supergroup information, refer to the content of our `meta_dict_pretrain.json` file, which serves as a template for setting the additional information. If you are training a model from scratch, you have the flexibility to define your own "prop" and "prop_id" values for the task's supergroup. However, if you intend to utilize our pretrained model, it is crucial to use the `meta_dict_pretrain.json` file when setting the `--task-dict-dir` argument to ensure compatibility and consistent results.


## Training
To start training, run the following command:
```bash
fairseq-train --user-dir ./graphormer [your/directory/to/the/lmdb/folder] --mixing-dataset --valid-subset val_id --best-checkpoint-metric R2_acc_mean --maximize-best-checkpoint-metric --num-workers 0 --task dft_md_combine --criterion mae_dft_md --arch IEFormer_ep_pp_dft_md --optimizer adam --adam-betas 0.9,0.98 --adam-eps 1e-6 --clip-norm 2 --lr-scheduler polynomial_decay --lr 1e-5 --warmup-updates 5000 --total-num-update 500000 --batch-size 8 --dropout 0.0 --attention-dropout 0.1 --weight-decay 0.001 --update-freq 1 --seed 1 --wandb-project DRFormer_ADMET --embed-dim 768 --ffn-embed-dim 768 --attention-heads 48 --max-update 500000 --log-interval 100 --log-format simple --save-interval 2 --validate-interval-updates 1 --keep-interval-updates 20 --save-dir [your/directory/to/save/the/checkpoints/and/reslts] --layers 12 --blocks 4 --required-batch-size-multiple 1 --node-loss-weight 1 --use-fit-sphere --use-shift-proj --edge-loss-weight 1 --sphere-pass-origin --use-unnormed-node-label --noisy-nodes --noisy-nodes-rate 1.0 --noise-scale 0.2 --noise-type normal --noise-in-traj --noisy-node-weight 1 --SAA-idx 0 --explicit-pos --pos-update-freq 6 --drop-or-add --cls-weight 1 --deform-tail --mix-reg-cls --neg-inf-before-softmax --readout-attention --moe 8 --task-dict-dir ./meta_dict.json --moe-in-backbone --ddp-backend legacy_ddp --drop-tail --task-type-num 90 --use-meta --data-balance 0.2
```

### Example Command
Here is an example command to train the model using data stored in the `example/lmdb` directory:
```bash
fairseq-train --user-dir ./graphormer example/lmdb --mixing-dataset --valid-subset val_id --best-checkpoint-metric R2_acc_mean --maximize-best-checkpoint-metric --num-workers 0 --task dft_md_combine --criterion mae_dft_md --arch IEFormer_ep_pp_dft_md --optimizer adam --adam-betas 0.9,0.98 --adam-eps 1e-6 --clip-norm 2 --lr-scheduler polynomial_decay --lr 1e-5 --warmup-updates 5000 --total-num-update 500000 --batch-size 8 --dropout 0.0 --attention-dropout 0.1 --weight-decay 0.001 --update-freq 1 --seed 1 --wandb-project DRFormer_ADMET --embed-dim 768 --ffn-embed-dim 768 --attention-heads 48 --max-update 500000 --log-interval 100 --log-format simple --save-interval 2 --validate-interval-updates 1 --keep-interval-updates 20 --save-dir ./checkpoints/example --layers 12 --blocks 4 --required-batch-size-multiple 1 --node-loss-weight 1 --use-fit-sphere --use-shift-proj --edge-loss-weight 1 --sphere-pass-origin --use-unnormed-node-label --noisy-nodes --noisy-nodes-rate 1.0 --noise-scale 0.2 --noise-type normal --noise-in-traj --noisy-node-weight 1 --SAA-idx 0 --explicit-pos --pos-update-freq 6 --drop-or-add --cls-weight 1 --deform-tail --mix-reg-cls --neg-inf-before-softmax --readout-attention --moe 8 --task-dict-dir ./meta_dict.json --moe-in-backbone --ddp-backend legacy_ddp --drop-tail --task-type-num 90 --use-meta --data-balance 0.2
```

### Pretrained model
We provide a pretrained model for evaluating the performance of our approach on ADMET prediction tasks. You can download the pretrained model checkpoint file checkpoint_OmniMol_ADMET.pt from the following OneDrive link:
[Download Pretrained Model "checkpoint_OmniMol_ADMET.pt"](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155156871_link_cuhk_edu_hk/Ee8rTDta3E9Gm8aSXquqZDUBiuYPc3T_P0JN-fV_SC-xcQ?e=vjPemY)

To evaluate the performance of our pretrained model on your own ADMET prediction, prepare the targeted ADMET dataset following the above instructions. and Run the following command by replace [your/directory/to/the/ADMET/lmdb/folder] with the directory path where your ADMET dataset in LMDB format is located:

```bash
fairseq-train --user-dir ./graphormer [your/directory/to/the/ADMET/lmdb/folder] --mixing-dataset --valid-subset test_id --best-checkpoint-metric R2_acc_mean --maximize-best-checkpoint-metric --num-workers 0 --task dft_md_combine --criterion mae_dft_md --arch IEFormer_ep_pp_dft_md --optimizer adam --adam-betas 0.9,0.98 --adam-eps 1e-6 --clip-norm 2 --lr-scheduler polynomial_decay --lr 0 --warmup-updates 5000 --total-num-update 1 --batch-size 8 --dropout 0.0 --attention-dropout 0.1 --weight-decay 0.001 --update-freq 4 --seed 1 --wandb-project DRFormer_ADMET --embed-dim 768 --ffn-embed-dim 768 --attention-heads 48 --max-update 1 --log-interval 100 --log-format simple --save-interval 2 --validate-interval-updates 1 --keep-interval-updates 20 --save-dir ./checkpoints/ADMET_eval --layers 12 --blocks 4 --required-batch-size-multiple 1 --node-loss-weight 1 --use-fit-sphere --use-shift-proj --edge-loss-weight 1 --sphere-pass-origin --use-unnormed-node-label --noisy-nodes --noisy-nodes-rate 1.0 --noise-scale 0.2 --noise-type normal --noise-in-traj --noisy-node-weight 1 --SAA-idx 0 --explicit-pos --pos-update-freq 6 --drop-or-add --cls-weight 1 --deform-tail --mix-reg-cls --neg-inf-before-softmax --readout-attention --moe 8 --task-dict-dir ./meta_dict.json --moe-in-backbone --ddp-backend legacy_ddp --drop-tail --task-type-num 90 --use-meta --data-balance 0.2 --restore-file [your/directory/to/our/pretrained/model]/checkpoint_OmniMol_ADMET.pt --reset-dataloader --reset-lr-scheduler --reset-optimizer --reset-meters --distributed-world-size 1 --device-id 0
```

Example: to demonstrate the functionality of our pretrained model, we provide an example dataset named `CYP2C19-inh`. To evaluate the model's performance on this dataset, execute the following command:

```bash
fairseq-train --user-dir ./graphormer example/lmdb --mixing-dataset --valid-subset test_id --best-checkpoint-metric R2_acc_mean --maximize-best-checkpoint-metric --num-workers 0 --task dft_md_combine --criterion mae_dft_md --arch IEFormer_ep_pp_dft_md --optimizer adam --adam-betas 0.9,0.98 --adam-eps 1e-6 --clip-norm 2 --lr-scheduler polynomial_decay --lr 0 --warmup-updates 5000 --total-num-update 1 --batch-size 8 --dropout 0.0 --attention-dropout 0.1 --weight-decay 0.001 --update-freq 4 --seed 1 --wandb-project DRFormer_ADMET --embed-dim 768 --ffn-embed-dim 768 --attention-heads 48 --max-update 1 --log-interval 100 --log-format simple --save-interval 2 --validate-interval-updates 1 --keep-interval-updates 20 --save-dir ./checkpoints/ADMET_eval --layers 12 --blocks 4 --required-batch-size-multiple 1 --node-loss-weight 1 --use-fit-sphere --use-shift-proj --edge-loss-weight 1 --sphere-pass-origin --use-unnormed-node-label --noisy-nodes --noisy-nodes-rate 1.0 --noise-scale 0.2 --noise-type normal --noise-in-traj --noisy-node-weight 1 --SAA-idx 0 --explicit-pos --pos-update-freq 6 --drop-or-add --cls-weight 1 --deform-tail --mix-reg-cls --neg-inf-before-softmax --readout-attention --moe 8 --task-dict-dir ./meta_dict_pretrain.json --moe-in-backbone --ddp-backend legacy_ddp --drop-tail --task-type-num 90 --use-meta --data-balance 0.2 --restore-file [your/directory/to/our/pretrained/model]/checkpoint_OmniMol_ADMET.pt --reset-dataloader --reset-lr-scheduler --reset-optimizer --reset-meters --distributed-world-size 1 --device-id 0
```

The evaluation process typically takes no more than 5 minutes on a server equipped with a V100 GPU. Upon completion, the evaluation results will be saved in the file `./checkpoints/ADMET_eval/record/cls_task_step1_epoch1.csv`. The expected performance metrics for this example dataset are a ROC-AUC of `0.924` and an accuracy of `0.857`.

## Finetuning
If further finetuning for better ADMET prediction performance is needed for better performance, please refer to `ADMET_FT_cls.sh` and `ADMET_FT_reg.sh` for specific scripts designed for finetuning the model. Make sure to adjust the scripts accordingly to fit your dataset directory and training requirements.

Here are the example commands to start finetuning:

### For Classification Tasks
```bash
bash ADMET_FT_cls.sh 
```

### For Regression Tasks
```bash
bash ADMET_FT_reg.sh 
```

These scripts assume you have pre-defined settings and configurations suitable for your specific ADMET prediction tasks. Adjust the parameters such as learning rate, batch size, and number of epochs according to your dataset characteristics and computational resources.

##  Computational Resources
The pretraining of our model on the full combination of all ADMET datasets was performed on 4 NVIDIA A100-80GB GPUs for approximately 90 hours.
Finetuning on specific ADMET datasets typically takes around 3 hours on a single NVIDIA A100-80GB GPU.
