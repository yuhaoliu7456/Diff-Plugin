import torch
from cleanfid import fid
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--restored', '--res', type=str, default=None)
parser.add_argument('--reference', '--gt', type=str, default=None)
parser.add_argument('--t', type=str, default=None, required=True, 
                    choices=['rain', 'snow', 'haze', 'blur', 'face', 'highlight', 'moire', 'lowlight'],
                    help='task name')


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

args = parser.parse_args()
restored_dir = args.restored
reference_path = args.reference


print(f'-----restored_dir: {restored_dir}')
print(f'-----reference_path: {reference_path}')

fid_score = fid.compute_fid(restored_dir, dataset_name=args.t, model_name="clip_vit_b_32", mode="clean", dataset_split="custom")
print(f"-----CLIP-FID: {fid_score:.5f}\n\n")