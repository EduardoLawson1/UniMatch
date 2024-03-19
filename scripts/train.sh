#o que eu usei aqui para mudar:
#https://chat.openai.com/c/2ff9b698-2421-44b2-b3e2-bdca21de3db1
#https://pytorch.org/docs/stable/elastic/run.html

#!/bin/bash
export RANK=0
export WANDB_API_KEY="379a9e3e59d4b4cd151585beb228abeadc8404cb"
export CUDA_LAUNCH_BLOCKING=1
now=$(date +"%Y%m%d_%H%M%S")

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['pascal', 'cityscapes', 'coco']
# method: ['unimatch', 'fixmatch', 'supervised']
# exp: just for specifying the 'save_path'
# split: ['92', '1_16', 'u2pl_1_16', ...]. Please check directory './splits/$dataset' for concrete splits
dataset='pascal'
method='unimatch'
exp='r101'
split='92'

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split

mkdir -p $save_path

torchrun \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.log


#python -m torch.distributed.launch \
    #--nproc_per_node=$1 \
    #--master_addr=localhost \
    #--master_port=$2 \
    #$method.py \
    #--config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    #--save-path $save_path --port $2 2>&1 | tee $save_path/$now.log
