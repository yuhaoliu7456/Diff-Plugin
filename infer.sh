task=face # desnow, dehaze, derain, deblur, demoire, highlight, lowlight, face

python infer.py \
    --img_path data/val/$task/sample1.jpg \
    --ckpt_dir pre-trained/$task/ \
    --num_inference_steps 20 \
    --pretrained_model_name_or_path "/mnt/ryn01/Checkpoints/models--CompVis--stable-diffusion-v1-4" \  # either a path to a directory containing model weights or a model identifier (e.g., CompVis/stable-diffusion-v1-4) from huggingface.co/models
    # --inp_of_unet_is_random_noise    # this line is specific for lowlight and highlight tasks
