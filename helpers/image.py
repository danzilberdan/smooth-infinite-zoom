from PIL import Image
import requests
import numpy as np
import math

def image_grid(imgs, rows, cols):
  assert len(imgs) == rows*cols

  w, h = imgs[0].size
  grid = Image.new('RGB', size=(cols*w, rows*h))
  grid_w, grid_h = grid.size

  for i, img in enumerate(imgs):
      grid.paste(img, box=(i%cols*w, i//cols*h))
  return grid

sampling_ratio = 0.25

def find_similar_coordinates(image1, image2):
    # Calculate the average color of image1
    image1_rgb = image1.convert("RGB")
    image1_width, image1_height = image1_rgb.size
    total_pixels = image1_width * image1_height
    image1_pixels = image1_rgb.load()

    r_sum = g_sum = b_sum = 0
    for y in range(image1_height):
        for x in range(image1_width):
            r, g, b = image1_pixels[x, y]
            r_sum += r
            g_sum += g
            b_sum += b

    avg_r = int(r_sum / total_pixels)
    avg_g = int(g_sum / total_pixels)
    avg_b = int(b_sum / total_pixels)

    # Find coordinates in image2 with the most similar colors to the average of image1
    image2_rgb = image2.convert("RGB")
    image2_width, image2_height = image2_rgb.size
    image2_pixels = image2_rgb.load()

    best_coordinates = []
    min_color_diff = float("inf")

    for y in range(0, image2_height - image1_height + 1, round(sampling_ratio * image1_height)):
        for x in range(0, image2_width - image1_width + 1, round(sampling_ratio * image1_width)):
            total_diff = 0

            for j in range(image1_height):
                for i in range(image1_width):
                    r2, g2, b2 = image2_pixels[x + i, y + j]
                    diff_r = abs(r2 - avg_r)
                    diff_g = abs(g2 - avg_g)
                    diff_b = abs(b2 - avg_b)
                    total_diff += diff_r + diff_g + diff_b

            print(f'Current best is {min_color_diff} and this options is {total_diff}.')
            if total_diff < min_color_diff:
                min_color_diff = total_diff
                best_coordinates = [(x, y)]
            elif total_diff == min_color_diff:
                best_coordinates.append((x, y))

    return best_coordinates[0]

def slices_around_coords(coords, height, width, max_x, max_y):
    lower_x = max(coords[0] - round(height / 2), 0)
    upper_x = min(round(lower_x + height), max_x)
    lower_x = upper_x - height

    lower_y = max(coords[1] - round(width / 2), 0)
    upper_y = min(round(lower_y + width), max_y)
    lower_y = upper_y - width

    return [slice(round(lower_x), round(upper_x)), slice(round(lower_y), round(upper_y))]

def shrink_and_paste_on_blank(current_image, mask_width):
  """
  Decreases size of current_image by mask_width pixels from each side,
  then adds a mask_width width transparent frame, 
  so that the image the function returns is the same size as the input. 
  :param current_image: input image to transform
  :param mask_width: width in pixels to shrink from each side
  """

  height = current_image.height
  width = current_image.width

  #shrink down by mask_width
  prev_image = current_image.resize((height-2*mask_width,width-2*mask_width))
  prev_image = prev_image.convert("RGBA")
  prev_image = np.array(prev_image)

  #create blank non-transparent image
  blank_image = np.array(current_image.convert("RGBA"))*0
  blank_image[:,:,3] = 1

  #paste shrinked onto blank
  blank_image[mask_width:height-mask_width,mask_width:width-mask_width,:] = prev_image
  prev_image = Image.fromarray(blank_image)

  return prev_image
  
def load_img(address, res=(512, 512)):
    if address.startswith('http://') or address.startswith('https://'):
        image = Image.open(requests.get(address, stream=True).raw)
    else:
        image = Image.open(address)
    image = image.convert('RGB')
    image = image.resize(res, resample=Image.LANCZOS)
    return image