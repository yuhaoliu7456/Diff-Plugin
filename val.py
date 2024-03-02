import os
import torch
import numpy as np
from diffusers.utils import load_image
from diffusers import UniPCMultistepScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPImageProcessor
from utils import get_latest_checkpoint, load_states, concat_imgs


def log_validation(logger, vae, unet, image_encoder,  scb_net, tpb_net,  args, accelerator, step):
    logger.info("Running validation... ")

    scb_net = accelerator.unwrap_model(scb_net)
    tpb_net = accelerator.unwrap_model(tpb_net)
    noise_scheduler = UniPCMultistepScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    image_logs = validation(args, vae, unet, image_encoder,  scb_net, tpb_net, noise_scheduler, args.validation_image, logger=logger)

    for tracker in accelerator.trackers:
        img_list_for_save = []
        for log in image_logs:
            pred = log["pred"]
            inp = log["input"]
            img_list_for_save += [pred]
            formatted_images = []

            formatted_images.append(np.asarray(inp.resize((pred.size))))
            formatted_images.append(np.asarray(pred))
            formatted_images = np.stack(formatted_images)
            tracker.writer.add_images("null", formatted_images, step, dataformats="NHWC")

    concat_imgs(img_list_for_save).save(os.path.join(args.output_dir, "visuals", str(step) + '.jpg'), target_size=512, target_dim=1)

    return image_logs



def validation(args, vae, unet, image_encoder, scb_net, tpb_net, noise_scheduler, validation_images, logger=None):
    
    image_logs = []
    path = get_latest_checkpoint(args.output_dir)
    states_path = os.path.join(args.output_dir, path, "random_states_0.pkl")
    
    for index, validation_image in enumerate(validation_images):

        image = load_image(validation_image)
        clip_image_processor = CLIPImageProcessor()
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=True)
        scb_net.eval()
        tpb_net.eval()

        with torch.no_grad():
            clip_visual_input = clip_image_processor(images=image, return_tensors="pt").pixel_values.to(device=vae.device)
            prompt_embeds = tpb_net(clip_vision_outputs=image_encoder(clip_visual_input, output_attentions=True, output_hidden_states=True),
                                                        use_global=args.used_clip_vision_global,
                                                        layer_ids=args.used_clip_vision_layers,)

            load_states(states_path)

            latents = torch.randn((1,4, 64, 64)).cuda() 
            noise_scheduler.set_timesteps(args.num_inference_steps, device=vae.device)
            timesteps = noise_scheduler.timesteps
            timesteps = timesteps.long()

            image = vae_image_processor.preprocess(image, height=args.resolution, width=args.resolution).to(device=vae.device)  # image now is tensor in [-1,1]

            SCB_cond = vae.config.scaling_factor * torch.chunk(vae.quant_conv(vae.encoder(image)), 2, dim=1)[0]

            for _, t in enumerate(timesteps):
                
                down_block_res_samples = scb_net(latents, t, encoder_hidden_states=prompt_embeds, cond_img=SCB_cond, return_dict=False,)
                noise_pred = unet(latents, t, encoder_hidden_states=prompt_embeds, down_block_additional_residuals=down_block_res_samples).sample

                latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]


            pred = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
            pred_pil = vae_image_processor.postprocess(pred, output_type='pil')[0]
            

            image_logs.append({"input": load_image(validation_image), "pred": pred_pil,})

    return image_logs
