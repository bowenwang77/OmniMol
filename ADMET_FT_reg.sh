
##Regression

datasets=(
    IGC50
    LC50
    LC50DM
    LogD
    LogP
    LogS
    MDCK
    BCF
    Fu
    PPB
    VDss
    Caco-2
)

# Command Template
run_command() {
    dataset=$1
    device_id=$2
    fairseq-train --user-dir ./graphormer \
        [your/directory/to/the/ADMET/lmdb/folder]/$dataset \
        --valid-subset test_id \
        --best-checkpoint-metric R2 --maximize-best-checkpoint-metric --num-workers 0 --task dft_md_combine --criterion mae_dft_md --arch IEFormer_ep_pp_dft_md \
        --optimizer adam --adam-betas 0.9,0.98 \
        --adam-eps 1e-6 --clip-norm 2 --lr-scheduler polynomial_decay --lr 1e-5 --warmup-updates 500 \
        --total-num-update 20000 --batch-size 8 \
        --dropout 0.0 --attention-dropout 0.1 \
        --weight-decay 0.001 --update-freq 1 --seed 1 --wandb-project DRFormer_ADMET --embed-dim 768 --ffn-embed-dim 768 --attention-heads 48 \
        --max-update 20000 --log-interval 100 --log-format simple --save-interval 3 --validate-interval 3 --keep-interval-updates 5 \
        --no-epoch-checkpoints \
        --save-dir [your/saving/directory]/FT_$dataset \
        --layers 12 --blocks 4 \
        --required-batch-size-multiple 1 --node-loss-weight 1 --use-fit-sphere --use-shift-proj --edge-loss-weight 1 --sphere-pass-origin \
        --use-unnormed-node-label --noisy-nodes --noisy-nodes-rate 1.0 --noise-scale 0.2 --noise-type normal --noise-in-traj \
        --noisy-node-weight 1 --SAA-idx 0 --explicit-pos --pos-update-freq 6 --drop-or-add --cls-weight 5 --deform-tail \
        --mix-reg-cls --neg-inf-before-softmax --readout-attention --moe 8 \
        --task-dict-dir [your/directory/to]/meta_dict.json --moe-in-backbone --ddp-backend legacy_ddp \
        --task-type-num 90 --use-meta \
        --distributed-world-size 1 \
        --device-id $device_id \
        --restore-file [your/directory/to]/checkpoint_OmniMol_ADMET.pt \
        --reset-dataloader --reset-lr-scheduler --reset-optimizer --reset-meters 
}

export -f run_command

# Command Execution
counter=0
for dataset in "${datasets[@]}"; do
    run_command $dataset $((counter % 1)) &
    let "counter++"
    # Ensure we only run 2 jobs in parallel
    if [[ $((counter % 2)) -eq 1 ]]; then
        wait
    fi
done

# Wait for any remaining jobs
wait