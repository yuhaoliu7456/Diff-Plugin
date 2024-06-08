from cleanfid import fid


# Take the demoire as an example,

custom_name = 'moire'
dataset_path = 'YOUR_ROOT_PATH/Demoireing/AIM19demoire/Training/clear'
fid.make_custom_stats(custom_name, dataset_path, mode="clean", model_name="clip_vit_b_32")


# custom_name = 'rain'
# dataset_path = 'YOUR_ROOT_PATH/DeRain/SPA+/real_world_gt'
# fid.make_custom_stats(custom_name, dataset_path, mode="clean", model_name="clip_vit_b_32")


# custom_name = 'snow'
# dataset_path = 'YOUR_ROOT_PATH/Dataset/DeSnow/RealSnow/Real_Snow_training_original_size/video2imgs_GT_re/video2imgs_GT_re'
# fid.make_custom_stats(custom_name, dataset_path, mode="clean", model_name="clip_vit_b_32")

# custom_name = 'haze'
# dataset_path = 'YOUR_ROOT_PATH/DeHaze/Reside/outdoor/train/clear/'
# fid.make_custom_stats(custom_name, dataset_path, mode="clean", model_name="clip_vit_b_32")


# custom_name = 'blur'
# dataset_path = 'YOUR_ROOT_PATH/DeBlur/Gopro/train/sharp/'
# fid.make_custom_stats(custom_name, dataset_path, mode="clean", model_name="clip_vit_b_32")


# custom_name = 'face'
# dataset_path = 'YOUR_ROOT_PATH/Face/FFHQ512/'
# fid.make_custom_stats(custom_name, dataset_path, mode="clean", model_name="clip_vit_b_32")


# custom_name = 'highlight'
# dataset_path = 'YOUR_ROOT_PATH/Highlight/SHIQ_data_10825/train_gt/'
# fid.make_custom_stats(custom_name, dataset_path, mode="clean", model_name="clip_vit_b_32")


# custom_name = 'lowlight'
# dataset_path = 'YOUR_ROOT_PATH/LowLight/LOL/our485/high'
# fid.make_custom_stats(custom_name, dataset_path, mode="clean", model_name="clip_vit_b_32")
