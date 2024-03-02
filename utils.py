import os
import torch
import random
import numpy as np
import time
import shutil
from PIL import Image

from transformers import PretrainedConfig




def save_args(args):
    argsDict = args.__dict__
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.txt'), 'a+') as f:
        f.writelines('------------------start at time:' + time.asctime( time.localtime(time.time()) ) + '------------------'+ '\n')
        for key in argsDict.keys():
            f.write(key + ':' + str(argsDict[key]) + '\n')
        f.writelines('------------------end at time:' + time.asctime( time.localtime(time.time()) ) + '------------------'+ '\n')

def code_backup(args):
    code_path = os.path.join(args.output_dir, 'code_backup')
    if os.path.exists(code_path):
        shutil.rmtree(code_path)
    ignore_patterns = shutil.ignore_patterns("*.pyc", "__pycache__", "results", "cache", "*.csv", "slurm_logs", "wandb", "temp_results", "*.log")
    shutil.copytree(args.project_path, code_path, ignore=ignore_patterns)



def save_random_states(logger, output_dir, process_index=0):
    from accelerate.utils import RNG_STATE_NAME
    # Random states
    # Random number generator states
    states = {}
    states_name = f"{RNG_STATE_NAME}_{process_index}.pkl"
    states["random_state"] = random.getstate()
    states["numpy_random_seed"] = np.random.get_state()
    states["torch_manual_seed"] = torch.get_rng_state()
    states["torch_cuda_manual_seed"] = torch.cuda.get_rng_state_all()
    output_states_file = os.path.join(output_dir, states_name)
    torch.save(states, output_states_file)
    logger.info(f"Random states saved in {output_states_file}")


def get_latest_checkpoint(output_dir, prefix='checkpoint'):
    # Get the most recent checkpoint
    # by default, the checkpoint is saved as checkpoint-<step>
    dirs = os.listdir(output_dir)
    dirs = [d for d in dirs if d.startswith(prefix)]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    path = dirs[-1] if len(dirs) > 0 else None
    return path




def load_states(states_path):
    # Code from accelerate.checkpointing.py Liine 185-188
    # the random states are must loaded from the training process, otherwise the results will be different
    states = torch.load(states_path)
    random.setstate(states["random_state"])
    np.random.set_state(states["numpy_random_seed"])
    torch.set_rng_state(states["torch_manual_seed"])
    # torch.cuda.set_rng_state_all(states["torch_cuda_manual_seed"])


def concat_imgs(imgs, target_size=(512,512), target_dim=1):
    # imgs is a list of PIL images
    # the two images should be the smae format; return is PIL image
    assert [type(imgs[0]) == type(imgs[i]) for i in range(len(imgs))]
    # for now, just handle PIL image
    imgs = [np.array(image.resize(target_size)) for image in imgs]
    save_ = np.concatenate(imgs, axis=target_dim)
    save_ = Image.fromarray(save_)
    return save_





def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")



def _encode_prompt(
        prompt,
        device,
        tokenizer,
        text_encoder,
        num_images_per_prompt=1,
        use_global_text=False

):
    # copied from diffuser.ControNet

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
    ):
        removed_text = tokenizer.batch_decode(
            untruncated_ids[:, tokenizer.model_max_length - 1: -1]
        )
        print(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {tokenizer.model_max_length} tokens: {removed_text}"
        )

    if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
        attention_mask = text_inputs.attention_mask.to(device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids.to(device),
        attention_mask=attention_mask,
    )

    prompt_embeds = prompt_embeds[0] if not use_global_text else prompt_embeds[1].unsqueeze(1) # [1,768]->[1,1,768]

    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    return prompt_embeds