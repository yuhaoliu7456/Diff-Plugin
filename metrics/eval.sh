export CUDA_VISIBLE_DEVICES=0

# take the demoire task as an example

python fid.py --t moire --restored  YOUR_RESULTS_PATH/demoireing/AIM19
python kid.py --t moire --restored  YOUR_RESULTS_PATH/demoireing/AIM19