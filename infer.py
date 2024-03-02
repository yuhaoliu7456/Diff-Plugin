import os
import argparse
import torch

from transformers import CLIPVisionModel, AutoTokenizer, CLIPImageProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from diffusers.image_processor import VaeImageProcessor

from modules import SCBNet
from modules import TPBNet
from utils import concat_imgs, import_model_class_from_model_name_or_path






def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a Diff-Plugin inference script.")

    parser.add_argument("--pretrained_model_name_or_path", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--clip_path",default="openai/clip-vit-large-patch14")
    parser.add_argument("--inp_of_crossatt", type=str, default='clip', choices=['text', 'clip'],)
    parser.add_argument("--inp_of_unet_is_random_noise", action="store_true", default=False, help="only set this to True for lowlight and highlight tasks")

    parser.add_argument("--ckpt_dir", type=str, default="", required=False,)
    parser.add_argument("--used_clip_vision_layers", type=int, default=24,)
    parser.add_argument("--used_clip_vision_global", action="store_true", default=False,)
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--num_inference_steps", type=int, default=20,)
    parser.add_argument("--time_threshold", type=int, default=960, help='this is used when we set the initial noise as inp+noise')
    parser.add_argument("--save_root", default="temp_results/", )
    
    parser.add_argument("--seed", type=int, default=42,)
    parser.add_argument("--img_path", type=str, required=True)


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    return args

if __name__ == "__main__":

    args = parse_args()

    # step-1: settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SCBNet_path = os.path.join(args.ckpt_dir, "scb") 
    TPBNet_path = os.path.join(args.ckpt_dir, "tpb.pt")
    print('--------loading SCB from: ', SCBNet_path, '   , TPB from:  ', TPBNet_path, '----------------------')
    os.makedirs(args.save_root, exist_ok=True)

    
    # Step-2: instantiate models and schedulers
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=None).to(device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=None).to(device)
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, None)
    text_encoder = text_encoder_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=None).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=None, use_fast=False,)
    clip_v = CLIPVisionModel.from_pretrained(args.clip_path).to(device)
    noise_scheduler = UniPCMultistepScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    clip_image_processor = CLIPImageProcessor()
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=True)
    
    scb_net = SCBNet.from_pretrained(SCBNet_path).to(device)
    tpb_net = TPBNet().to(device) 
    try:
        tpb_net.load_state_dict(torch.load(TPBNet_path)['model'], strict=True)
    except:
        tpb_net = torch.nn.DataParallel(tpb_net)
        tpb_net.load_state_dict(torch.load(TPBNet_path)['model'], strict=True)
 
    scb_net.eval()
    tpb_net.eval()


    # Step-3: prepare data
    image = load_image(args.img_path)
    pil_image = image.copy()
    

    with torch.no_grad():
        # TPB
        clip_visual_input = clip_image_processor(images=image, return_tensors="pt").pixel_values.to(device=vae.device)
        prompt_embeds = tpb_net(clip_vision_outputs=clip_v(clip_visual_input, output_attentions=True, output_hidden_states=True),
                                use_global=args.used_clip_vision_global,
                                layer_ids=args.used_clip_vision_layers,)

        # resolution adjustment (one can adjust this resolution also, as long as the short side is equal to or larger than 512)
        width, height = image.size
        if width < 512 or height < 512:
            if width < height:
                new_width = 512
                new_height = int((512 / width) * height)
            else:
                new_height = 512
                new_width = int((512 / height) * width)
            image = image.resize((new_width, new_height))
        else:
            new_height = height
            new_width = width

        
        # pre-process image
        image = vae_image_processor.preprocess(image, height=new_height, width=new_width).to(device=vae.device)  # image now is tensor in [-1,1]
        scb_cond = vae.config.scaling_factor * torch.chunk(vae.quant_conv(vae.encoder(image)), 2, dim=1)[0]
        b, c, h, w = scb_cond.size()

        # set/load random seed
        generator = torch.Generator()
        generator.manual_seed(args.seed) # one can also adjust this seed to get different results

        # set the noise or latents
        if args.inp_of_unet_is_random_noise:
            latents = torch.randn((1,4, h, w), generator=generator).cuda()
        else:
            noise = torch.randn((1,4, h, w), generator=generator).cuda()

        # set the time step
        noise_scheduler.set_timesteps(args.num_inference_steps, device=vae.device)
        timesteps = noise_scheduler.timesteps
        timesteps = timesteps.long()

        # feedforward
        for i, t in enumerate(timesteps):
            # add noise 
            if t >= args.time_threshold and not args.inp_of_unet_is_random_noise:
                latents = noise_scheduler.add_noise(scb_cond, noise, t, )

            # SCB
            down_block_res_samples = scb_net(
                latents,
                t,
                encoder_hidden_states=prompt_embeds,
                cond_img=scb_cond,
                return_dict=False,
            )

            # diffusion unet
            noise_pred = unet(latents,
                t,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals= down_block_res_samples, 
            ).sample

            # update the latents
            latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # post-process
        pred = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
        pred = vae_image_processor.postprocess(pred, output_type='pil')[0]
    
    save_= concat_imgs([pil_image.resize(pred.size), pred], target_size=pred.size, target_dim=1)
    save_.save(os.path.join('./temp_results/', os.path.basename(args.img_path)))
    print('---------done-----------')
