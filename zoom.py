from helpers import *
from diffusers import StableDiffusionInpaintPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import gradio as gr
import numpy as np
import torch
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
inpaint_model_list = [
    "stabilityai/stable-diffusion-2-inpainting",
    "runwayml/stable-diffusion-inpainting",
    "parlance/dreamlike-diffusion-1.0-inpainting",
    "ghunkins/stable-diffusion-liberty-inpainting",
    "ImNoOne/f222-inpainting-diffusers"
]
default_prompt = "A psychedelic jungle with trees that have glowing, fractal-like patterns, Simon stalenhag poster 1920s style, street level view, hyper futuristic, 8k resolution, hyper realistic"
default_negative_prompt = "frames, borderline, text, charachter, duplicate, error, out of frame, watermark, low quality, ugly, deformed, blur"


def zoom(
    model_id,
    prompts_array,
    negative_prompt,
    num_outpainting_steps,
    guidance_scale,
    num_inference_steps,
    custom_init_image
):
    prompts = {}
    for x in prompts_array:
        try:
            key = int(x[0])
            value = str(x[1])
            prompts[key] = value
        except ValueError:
            pass
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config)
    pipe = pipe.to("cuda")

    def no_check(images, **kwargs):
        return images, False
    pipe.safety_checker = no_check
    pipe.enable_attention_slicing()
    g_cuda = torch.Generator(device='cuda')

    height = 512
    width = height

    current_image = Image.new(mode="RGBA", size=(height, width))
    mask_image = np.array(current_image)[:, :, 3]
    mask_image = Image.fromarray(255-mask_image).convert("RGB")
    current_image = current_image.convert("RGB")
    if (custom_init_image):
        current_image = custom_init_image.resize(
            (width, height), resample=Image.LANCZOS)
    else:
        init_images = pipe(prompt=prompts[min(k for k in prompts.keys() if k >= 0)],
                           negative_prompt=negative_prompt,
                           image=current_image,
                           guidance_scale=guidance_scale,
                           height=height,
                           width=width,
                           mask_image=mask_image,
                           num_inference_steps=num_inference_steps)[0]
        current_image = init_images[0]
    mask_width = 128
    num_interpol_frames = 30

    all_frames = []
    all_frames.append(current_image)

    for i in range(num_outpainting_steps):
        print('Outpaint step: ' + str(i+1) +
              ' / ' + str(num_outpainting_steps))

        prev_image_fix = current_image

        prev_image = shrink_and_paste_on_blank(current_image, mask_width)

        current_image = prev_image

        # create mask (black image with white mask_width width edges)
        mask_image = np.array(current_image)[:, :, 3]
        mask_image = Image.fromarray(255-mask_image).convert("RGB")

        # inpainting step
        current_image = current_image.convert("RGB")
        images = pipe(prompt=prompts[max(k for k in prompts.keys() if k <= i)],
                      negative_prompt=negative_prompt,
                      image=current_image,
                      guidance_scale=guidance_scale,
                      height=height,
                      width=width,
                      # generator = g_cuda.manual_seed(seed),
                      mask_image=mask_image,
                      num_inference_steps=num_inference_steps)[0]
        current_image = images[0]
        current_image.paste(prev_image, mask=prev_image)

        # interpolation steps bewteen 2 inpainted images (=sequential zoom and crop)
        for j in range(num_interpol_frames - 1):
            interpol_image = current_image
            interpol_width = round(
                (1 - (1-2*mask_width/height)**(1-(j+1)/num_interpol_frames))*height/2
            )
            interpol_image = interpol_image.crop((interpol_width,
                                                  interpol_width,
                                                  width - interpol_width,
                                                  height - interpol_width))

            interpol_image = interpol_image.resize((height, width))

            # paste the higher resolution previous image in the middle to avoid drop in quality caused by zooming
            interpol_width2 = round(
                (1 - (height-2*mask_width) / (height-2*interpol_width)) / 2*height
            )
            prev_image_fix_crop = shrink_and_paste_on_blank(
                prev_image_fix, interpol_width2)
            interpol_image.paste(prev_image_fix_crop, mask=prev_image_fix_crop)

            all_frames.append(interpol_image)
        all_frames.append(current_image)
        # interpol_image.show()
    video_file_name = "infinite_zoom_" + str(time.time())
    fps = 30
    save_path = video_file_name + ".mp4"
    start_frame_dupe_amount = 15
    last_frame_dupe_amount = 15

    write_video(save_path, all_frames, fps, False,
                start_frame_dupe_amount, last_frame_dupe_amount)
    return save_path


def zoom_app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():

                outpaint_prompts = gr.Dataframe(
                    type="array",
                    headers=["outpaint steps", "prompt"],
                    datatype=["number", "str"],
                    row_count=1,
                    col_count=(2, "fixed"),
                    value=[[0, default_prompt]],
                    wrap=True
                )

                outpaint_negative_prompt = gr.Textbox(
                    lines=1,
                    value=default_negative_prompt,
                    label='Negative Prompt'
                )

                outpaint_steps = gr.Slider(
                    minimum=5,
                    maximum=25,
                    step=1,
                    value=12,
                    label='Total Outpaint Steps'
                )
                with gr.Accordion("Advanced Options", open=False):
                    model_id = gr.Dropdown(
                        choices=inpaint_model_list,
                        value=inpaint_model_list[0],
                        label='Pre-trained Model ID'
                    )

                    guidance_scale = gr.Slider(
                        minimum=0.1,
                        maximum=15,
                        step=0.1,
                        value=7,
                        label='Guidance Scale'
                    )

                    sampling_step = gr.Slider(
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=50,
                        label='Sampling Steps for each outpaint'
                    )
                    init_image = gr.Image(type="pil")
                generate_btn = gr.Button(value='Generate video')

            with gr.Column():
                output_image = gr.Video(label='Output', format="mp4").style(
                    width=512, height=512)

        generate_btn.click(
            fn=zoom,
            inputs=[
                model_id,
                outpaint_prompts,
                outpaint_negative_prompt,
                outpaint_steps,
                guidance_scale,
                sampling_step,
                init_image
            ],
            outputs=output_image,
        )
