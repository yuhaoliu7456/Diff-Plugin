# Description: This is the training script for the Diff-Plugin-Release

# root setting
data_root='/mnt/ryn01/Dataset'
results_root='./'

# code setting 
MODEL_DIR="/mnt/ryn01/Checkpoints/models--CompVis--stable-diffusion-v1-4"
CLIP_PATH="openai/clip-vit-large-patch14"
project_path="./"
train_batch_size=1


# task setting
task="demoire" 
OUTPUT_DIR=$results_root"/results/"$task"/diff-plugin"

# accelerate launch --multi_gpu --num_processes=num_of_gpu train.py \         # this line is for multi-gpu training
python train.py \
 --project_path=$project_path \
 --data_root=$data_root \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --clip_path=$CLIP_PATH \
 --output_dir=$OUTPUT_DIR \
 --cache_dir=$results_root"/cache" \
 --train_data_file_list="data/train/$task.csv" \
 --tracker_project_name=$task \
 --scale_lr \
 --train_batch_size=$train_batch_size \
 --num_train_epochs=100 \
 --checkpointing_steps=10 \
 --validation_steps=10 \
 --validation_image "data/val/$task/sample1.jpg" \
 --allow_tf32 \
#  --use_data_aug