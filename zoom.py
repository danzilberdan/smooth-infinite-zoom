from typing import List
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

default_negative_prompt = "frames, borderline, text, charachter, duplicate, error, out of frame, watermark, low quality, ugly, deformed, blur"

height = 480
width = 720
num_interpol_frames = 30
small_ratio = 0.08
empty_ratio = 0.15

def zoom(
    model_id,
    negative_prompt,
    num_outpainting_steps,
    guidance_scale,
    num_inference_steps,
    files
):
    images = [Image.open(file.name) for file in files]
    
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

    frames = []

    small_image = images[0].resize(
        (width, height), resample=Image.LANCZOS).convert("RGBA")
    
    frames.append(small_image)

    for i in range(1, len(images)):
        print(f'Iteration {i}')
        
        outer_image = images[i].resize(
            (width, height), resample=Image.LANCZOS).convert("RGBA")
        
        resized_small = small_image.resize((round(width * small_ratio), round(height * small_ratio)))
        coords = np.array(find_similar_coordinates(resized_small, outer_image))
        print(f'The most similar coordinates found are: {coords}.')

        outer_image = np.array(outer_image)
        slices_for_small: List[slice] = slices_around_coords(coords, height * small_ratio, width * small_ratio, height, width)
        slices_for_empty: List[slice] = slices_around_coords(coords, height * empty_ratio, width * empty_ratio, height, width)

        # Updating coords in case the small image touches the border of the outer one.
        coords = np.array([round((slices_for_small[0].start + slices_for_small[0].stop) / 2),
                           round((slices_for_small[1].start + slices_for_small[1].stop) / 2)])

        outer_image[slices_for_empty[0], slices_for_empty[1], 3] = 1
        outer_image[slices_for_small[0], slices_for_small[1]] = np.array(resized_small)
        outer_image = Image.fromarray(outer_image)

        mask_image = np.array(outer_image)[:, :, 3]
        mask_image = Image.fromarray(255 - mask_image).convert("RGB")
        gen_image = pipe(prompt='',
                      negative_prompt=negative_prompt,
                      image=outer_image.convert("RGB"),
                      guidance_scale=guidance_scale,
                      height=height,
                      width=width,
                      mask_image=mask_image,
                      num_inference_steps=num_inference_steps)[0][0]
        
        middle = np.array([round(height / 2), round(width / 2)])

        for j in range(num_interpol_frames - 1):
            print(f'Rendering frame {j}.')
            current_middle = middle + (coords - middle) * (j + 1 / num_interpol_frames)
            # This value get's larger exponentially
            crop_size_ratio = round(
                small_ratio ** (1 - (j + 1) / num_interpol_frames)
            )
            raw_gen = np.array(gen_image)
            cropped_slices = slices_around_coords(current_middle, height * crop_size_ratio, width * crop_size_ratio, height, width)
            print(cropped_slices)
            cropped = Image.fromarray(raw_gen[cropped_slices[0], cropped_slices[1]])

            frame_base = cropped.resize((width, height))

            raw_frame_base = np.array(frame_base)
            insert_small_slices = slices_around_coords(
                current_middle, 
                height * crop_size_ratio * small_ratio, 
                width * crop_size_ratio * small_ratio, height, width)
            raw_frame_base[insert_small_slices[0], insert_small_slices[1]] = \
                    np.array(small_image.resize((round(width * crop_size_ratio * small_ratio), 
                                                 round(height * crop_size_ratio * small_ratio))))
            frame = Image.fromarray(raw_frame_base)

            frames.append(frame)
        frames.append(gen_image)



def zoom1(
    model_id,
    prompts_array,
    negative_prompt,
    num_outpainting_steps,
    guidance_scale,
    num_inference_steps,
    files
):
    images = [Image.open(file.name) for file in files]

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

    current_image = images[0].resize(
        (width, height), resample=Image.LANCZOS).convert("RGBA")

    mask_image = np.array(current_image)[:, :, 3]
    mask_image = Image.fromarray(255-mask_image).convert("RGB")
    current_image = current_image.convert("RGB")

    all_frames = []
    all_frames.append(current_image)

    for i in range(1, len(images)):
        print(f'Iteration {i}')
        
        prev_image_fix = current_image

        outer_image = images[i].resize(
            (width, height), resample=Image.LANCZOS).convert("RGBA")
        outer_image = np.array(outer_image)
        outer_image[outer_mask_width:height-outer_mask_width, outer_mask_width:width-outer_mask_width, 3] = 1
        outer_image = Image.fromarray(outer_image)

        prev_image = shrink_and_paste_on_blank(current_image, mask_width).convert("RGBA")
        prev_image.alpha_composite(outer_image)
        current_image = prev_image

        # create mask (black image with white mask_width width edges)
        mask_image = np.array(current_image)[:, :, 3]
        mask_image = Image.fromarray(255-mask_image).convert("RGB")

        # inpainting step
        current_image = current_image.convert("RGB")
        gen_images = pipe(prompt=prompts[max(k for k in prompts.keys() if k <= i)],
                      negative_prompt=negative_prompt,
                      image=current_image,
                      guidance_scale=guidance_scale,
                      height=height,
                      width=width,
                      # generator = g_cuda.manual_seed(seed),
                      mask_image=mask_image,
                      num_inference_steps=num_inference_steps)[0]
        current_image = gen_images[0]

        current_image.save('test.png', format='PNG')

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

                files = gr.File(file_types=["image"], file_count="multiple")

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
                        value=0.7,
                        label='Guidance Scale'
                    )

                    sampling_step = gr.Slider(
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=20,
                        label='Sampling Steps for each outpaint'
                    )
                generate_btn = gr.Button(value='Generate video')

            with gr.Column():
                output_image = gr.Video(label='Output', format="mp4").style(
                    width=width, height=height)

        generate_btn.click(
            fn=zoom,
            inputs=[
                model_id,
                outpaint_negative_prompt,
                outpaint_steps,
                guidance_scale,
                sampling_step,
                files
            ],
            outputs=output_image,
        )
