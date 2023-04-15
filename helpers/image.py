from PIL import Image
import requests
import numpy as np

def image_grid(imgs, rows, cols):
  assert len(imgs) == rows*cols

  w, h = imgs[0].size
  grid = Image.new('RGB', size=(cols*w, rows*h))
  grid_w, grid_h = grid.size

  for i, img in enumerate(imgs):
      grid.paste(img, box=(i%cols*w, i//cols*h))
  return grid

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