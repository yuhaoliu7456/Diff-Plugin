import os
import argparse
import torch
import gradio as gr

from transformers import CLIPVisionModel, AutoTokenizer, CLIPImageProcessor
from diffusers import AutoencoderKL, UniPCMultistepScheduler, UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor

from modules import TPBNet, SCBNet, PromptTunnerModel
from utils import _encode_prompt, import_model_class_from_model_name_or_path




def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a Diff-Plugin training script.")

    parser.add_argument("--pretrained_model_name_or_path", default="CompVis/stable-diffusion-v1-4",)
    parser.add_argument("--clip_path",default="openai/clip-vit-large-patch14",)
    parser.add_argument("--PromptTunner_path", type=str, default="pre-trained/plugin_selector/PromptTunner.pt", )

    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--num_inference_steps", type=int, default=20,)
    parser.add_argument("--time_threshold", type=int, default=960)
    parser.add_argument("--inp_of_unet_is_random_noise", action="store_true", default=False, help="this is used for lowlight and highlight tasks")

    parser.add_argument("--tunner_for_visual", type=bool, default=True, help='use this learnable tunner for visual projection')
    parser.add_argument("--use_global_text", type=bool, default=True, help='use the global text embedddings for visual projection')
    parser.add_argument("--used_clip_vision_layers", type=int, default=24,)
    parser.add_argument("--used_clip_vision_global", action="store_true", default=False,)
    parser.add_argument("--tasks", type=str, default=["derain","desnow","lowlight","dehaze","deblur","highlight","face","demoire"], nargs="+", help="task list.",)



    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()


    args.pretrained_model_name_or_path = '/mnt/ryn01/Checkpoints/models--CompVis--stable-diffusion-v1-4'
    args.clip_path = '/mnt/ryn01/Checkpoints/models--openai--clip-vit-large-patch14'

    return args



if __name__ == "__main__":

    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    # Step-2: instantiate models and schedulers
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=None).to(device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=None).to(device)
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, None)
    text_encoder = text_encoder_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=None).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=None, use_fast=False,)
    CLIP_V = CLIPVisionModel.from_pretrained(args.clip_path).to(device)

    noise_scheduler = UniPCMultistepScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    clip_image_processor = CLIPImageProcessor()
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=True)
    

    plugin_dict = {}
    for i, task in enumerate(args.tasks):
        tpb_net = TPBNet().to(device) 
        plug_in_file = os.path.join('pre-trained', task, "tpb.pt")
        assert os.path.exists(plug_in_file), plug_in_file + " does not exist"

        try:
            tpb_net.load_state_dict(torch.load(plug_in_file)['model'], strict=True)
        except:
            original_state_dict = torch.load(plug_in_file)['model']
            new_state_dict = {k.replace('module.', ''): v for k, v in original_state_dict.items()}
            tpb_net.load_state_dict(new_state_dict, strict=True)
        plugin_dict[task] = tpb_net



    PromptTunner = PromptTunnerModel(args=args).to(device)
    try:
        PromptTunner.load_state_dict(torch.load(args.PromptTunner_path)['model'], strict=True)
    except:
        PromptTunner = torch.nn.DataParallel(PromptTunner)
        PromptTunner.load_state_dict(torch.load(args.PromptTunner_path)['model'], strict=True)


    clip_image_processor = CLIPImageProcessor()
    vae_image_processor = VaeImageProcessor(vae_scale_factor=2 ** (len(vae.config.block_out_channels) - 1),
                                            do_convert_rgb=True,
                                            do_normalize=True)
    
    null_text_embeds = _encode_prompt(prompt=" ", device=vae.device, 
                                    tokenizer=tokenizer, text_encoder=text_encoder, 
                                    num_images_per_prompt=1, use_global_text=args.use_global_text)


    def process(image, prompt, hidden_state, inference_step=20, image_resolution=512, seed=42):
        """
        image: PIL image;
        prompt: str;
        """
        prompt = hidden_state[0] if hidden_state else prompt
        # the above line of code is equivalent to saying that the button is preferred if the button is clicked, otherwise the textbox text is used
        # for now, this gradio demo supports the case that the textbox is used and then the button is clicked
        # But, It does not support the case that the button is clicked and then the textbox is used !!!
        with torch.no_grad():
            # step-1: extract clip-t feats
            print('---------------prompt is : ', prompt, '   ----------------')
            clip_t_feats = _encode_prompt(prompt=prompt,
                                        device=vae.device,
                                        tokenizer=tokenizer,
                                        text_encoder=text_encoder,
                                        num_images_per_prompt=1,
                                        use_global_text=args.use_global_text)
            
            # step-2: extract clip-v feats
            clip_input = clip_image_processor(images=image, return_tensors="pt").pixel_values.to(device=vae.device)
            clip_v_feats = CLIP_V(clip_input, output_attentions=True, output_hidden_states=True)

            # step-3: get the plugin embeds
            plugin_gt_feats = {}
            plug_in_embeddings = torch.zeros(len(plugin_dict)+1, 256, 768).to(vae.device) 

            for idx, key in enumerate(plugin_dict.keys()):
                plugin_dict[key].eval()
                plugin_gt_feats[key] = plugin_dict[key](clip_vision_outputs=clip_v_feats, 
                                                            use_global=args.used_clip_vision_global, 
                                                            layer_ids=args.used_clip_vision_layers)
                plug_in_embeddings[idx] = plugin_gt_feats[key]
            
            plugin_gt_feats['null']=null_text_embeds
            if args.tunner_for_visual:
                plug_in_embeddings[-1] = null_text_embeds

            # setp-4: PromptTunner
            prompt_embeds = PromptTunner(clip_t_feats, plug_in_embeddings=plug_in_embeddings)
            
            # step-5: output the scores 
            similarity_score = prompt_embeds[:,0]
            similarity_score[similarity_score<0] = 0
            if similarity_score.any() == 0:
                similarity_score[-1]=1
            log_score = ' '.join([f"{key}: {round(value,2)}" for key, value in zip(plugin_gt_feats.keys(), similarity_score.tolist())])
            print('   Similarity scores are: ', log_score)


            # get all plugin index larger than threshold
            threshold = 0
            selected_index_list = torch.nonzero(similarity_score>threshold).squeeze().tolist()
            if not isinstance(selected_index_list, list):
                # for handling the case that int is returned
                selected_index_list = [selected_index_list]
            selected_plugin_key_list = [list(plugin_gt_feats.keys())[i] for i in selected_index_list]
            selected_plugin_embeds_list = [plugin_gt_feats[i] for i in selected_plugin_key_list]
            print('-------------------   selected Plugin index are: ', selected_plugin_key_list, '   ----------------')



            # ---------------------------------------------
            width, height = image.size
            if width < height:
                new_width = image_resolution
                new_height = int((image_resolution / width) * height)
            else:
                new_height = image_resolution
                new_width = int((image_resolution / height) * width)
            image = image.resize((new_width, new_height))


            
            for index, plugin_key in enumerate(selected_plugin_key_list):
                # loop over multiple plugins
                if plugin_key in ['lowlight', 'highlight']:
                    args.inp_of_unet_is_random_noise = True
                else:
                    args.inp_of_unet_is_random_noise = False


                if plugin_key == 'null':
                    pred = image
                    continue
                else:
                    prompt_embeds = selected_plugin_embeds_list[index]
                    plugin_control_path = os.path.join('pre-trained', plugin_key, "scb")
            

                print('Current image size is:  ', new_height, '  ' ,new_width,  '  Plugin is : ', plugin_key,  '  Control from:  ',plugin_control_path) 
                # use the selected plugin to process the image
                scb_net = SCBNet.from_pretrained(plugin_control_path).to(vae.device)
                

                image = vae_image_processor.preprocess(image, height=new_height, width=new_width).to(device=vae.device)  # image now is tensor in [-1,1]
                scb_cond = vae.config.scaling_factor * torch.chunk(vae.quant_conv(vae.encoder(image)), 2, dim=1)[0]
                b,c, h, w = scb_cond.size()

                generator = torch.Generator()
                generator.manual_seed(seed)

                if args.inp_of_unet_is_random_noise:
                    latents = torch.randn((1,4, h, w), generator=generator).cuda() #torch.randn_like(latents)
                else:
                    noise = torch.randn((1,4, h, w), generator=generator).cuda()

                # set the time step
                if index != 0:
                    # means that :
                    # 1. for the first plugin, we use the default inference_step
                    # 2. for the rest plugins, we use the much smaller inference_step to alleviate the noise accumulation
                    inference_step = 3

                noise_scheduler.set_timesteps(inference_step, device=vae.device)
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
                        encoder_hidden_states= prompt_embeds,
                        cond_img=scb_cond,
                        return_dict=False,
                    )
                    
                    # TPB
                    noise_pred = unet(
                        latents,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        down_block_additional_residuals=down_block_res_samples,
                    ).sample


                    latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                pred = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
                pred = vae_image_processor.postprocess(pred, output_type='pil')[0]
                
                if index != len(selected_plugin_key_list)-1:
                    image = pred.copy()  # this is used for the next plugin iteration

        return [pred]
    


    
    def set_snow_text(state):
        return "remove snow", state

    def set_rain_text(state):
        return "remove rain", state
    def set_haze_text(state):
        return "remove haze", state
    def set_blur_text(state):
        return "clear blur", state
    def set_face_text(state):
        return "enhance face appearance", state
    def set_low_text(state):
        return "enhance brightness", state
    def set_high_text(state):
        return "remove highlight", state
    def set_moire_text(state):
        return "I would like to remove the moire", state

    
    block = gr.Blocks().queue()
    with block:
        with gr.Row():
            gr.Markdown("## Diff-Plugin: Revitalizing Details for Diffusion-based Low-level Tasks")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(source='upload', type="pil").style(grid=2, height='auto')

            with gr.Column():
                result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')   
        
        with gr.Row():
            with gr.Column():
                # with gr.Accordion("Input Task Prompt", open=False):
                prompt = gr.Textbox(label="Prompt")
                hidden_state = gr.State()
            
            with gr.Column():
                with gr.Accordion("Select Task-Plugin", open=False):
                    with gr.Column():
                        with gr.Row():
                            desnow_button = gr.Button("Desnowing")
                            desnow_button.click(fn=set_snow_text, inputs=[hidden_state], outputs=[hidden_state])

                            dehaze_button = gr.Button("Dehazing")
                            dehaze_button.click(fn=set_haze_text, inputs=[hidden_state], outputs=[hidden_state])

                            face_button = gr.Button("Face Restoration")
                            face_button.click(fn=set_face_text, inputs=[hidden_state], outputs=[hidden_state])
                            
                            derain_button = gr.Button("Deraining")
                            derain_button.click(fn=set_rain_text, inputs=[hidden_state], outputs=[hidden_state])

                        with gr.Row():

                            lowlight_button = gr.Button("Lowlight Encement")
                            lowlight_button.click(fn=set_low_text, inputs=[hidden_state], outputs=[hidden_state])

                            deblur_button = gr.Button("Deblurring")
                            deblur_button.click(fn=set_blur_text, inputs=[hidden_state], outputs=[hidden_state])

                            highlight_button = gr.Button("Highlight Removal")
                            highlight_button.click(fn=set_high_text, inputs=[hidden_state], outputs=[hidden_state])

                            demoireing_button = gr.Button("Demoireing")
                            demoireing_button.click(fn=set_moire_text, inputs=[hidden_state], outputs=[hidden_state])
                            


            with gr.Column():
                run_button = gr.Button(label="Run")
                run_button.style(background_color='yellow', color='black')  
            
        with gr.Row():
            with gr.Accordion("Advanced options", open=False):
                image_resolution = gr.Slider(label="Image Resolution", minimum=512, maximum=1024, value=512, step=64)
                inference_step = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)

        ips = [input_image, prompt,hidden_state, inference_step, image_resolution, seed]
        run_button.click(fn=process, inputs=ips, outputs=[result_gallery])

        with gr.Row():
            with gr.Column():
                # add examples to your interface
                examples = [
                    ["data/demo/snow.jpg", "I want to remove the snow in this photo"],
                    ["data/demo/haze.png", "Can you help me clear the haze for this photo?"],
                ]
                gr.Examples(examples=examples, inputs=[input_image, prompt], outputs=[])
            
            with gr.Column():
                # add examples to your interface
                examples = [
                    ["data/demo/rain.jpg", "I want to remove rain in this image"],
                    ["data/demo/face.png", "I need to enhance the face appearance for this photo"],
                ]
                gr.Examples(examples=examples, inputs=[input_image, prompt], outputs=[])
            with gr.Column():
                # add examples to your interface
                examples = [
                    ["data/demo/lowlight.jpg", "I want to increase the brightness for this photo"],
                    ["data/demo/blur.jpg", "Could you guide me to eliminate the blurry element?"],
                ]
                gr.Examples(examples=examples, inputs=[input_image, prompt], outputs=[])
            
            with gr.Column():
                # add examples to your interface
                examples = [
                    ["data/demo/moire.png", "I would like to remove the moire"],
                    ["data/demo/highlight.png", "I want to clear the gleaming highlights"],
                ]
                gr.Examples(examples=examples, inputs=[input_image, prompt], outputs=[])
            
            with gr.Column():
                # add examples to your interface
                examples = [
                    ["data/demo/snow_and_haze.jpg", "remove snow"],
                    ["data/demo/snow_and_haze.jpg", "remove snow and haze"],
                ]
                gr.Examples(examples=examples, inputs=[input_image, prompt], outputs=[])
            
        with gr.Row():
            # add terms of use
            terms_of_use = """
            ---
            ### Terms of Use
            By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only.  It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.  We will collect those to keep improving our moderator. For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.

            ### License
            The service is a research preview intended for non-commercial use only, subject to the model [License](#) of Diff-Plugin and [Terms of Use](#) of the data generated by Stable Diffusion. Please contact us if you find any potential violation.
            
            ### Note:
            1. Different seed may lead to different results.
            2. If you encounter the issue that the result is very bad, please try to <restart> the gradio demo and try again. Especially when you change the prompt or the plugin, you maybe encounter this problem.
            3. The button and the textbox are not well connected, so if you first click the button and then try to use the textbox, please also try to restart the gradio demo and try again.

            """
            gr.Markdown(terms_of_use)



block.launch(server_name='0.0.0.0')



"""
Note that this gradio demo has some issues:
1. If you encounter the issue that the result is very bad, please try to restart the gradio demo and try again. Especially when you change the prompt or the plugin, you maybe encounter this problem.
2. The button and the textbox are not well connected, so if you first click the button and then try to use the textbox, please also try to restart the gradio demo and try again.
"""