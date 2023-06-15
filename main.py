from segment_anything import SamPredictor, sam_model_registry
from PIL import Image, ImageDraw
import numpy as np
import pydicom
from glob import glob


# open dicom file and convert into numpy array
# sag, cor, axi

def load_img():
    for dcm in glob('cor/*14.dcm'):
        ds = pydicom.dcmread(dcm)
        img = ds.pixel_array.astype(np.float32)
        img = 255 * (img - np.min(img)) / (np.max(img) - np.min(img))
        img = np.tile(img.astype(np.uint8)[:, :, np.newaxis], [1, 1, 3])
    return img  # numpy array with 3D shape (height, width, 3), uint8 ranged from 0 to 255


def overlay_mask(np_img, np_mask):
    mask = Image.fromarray(np_mask)
    overlay_img = Image.blend(Image.fromarray(np_img).convert('RGBA'), mask.convert('RGBA'), 0.5)
    return np.where(np_mask[:, :, 0:1] == 0, np_img, np.array(overlay_img.convert('RGB')))


img = load_img()
sam = sam_model_registry["vit_b"](checkpoint="./sam_hq_vit_b.pth")
predictor = SamPredictor(sam)
predictor.set_image(img)
coord = [204, 435, 687, 718]
# coord = [148, 419, 316, 553]
masks, _, _ = predictor.predict(box=np.array(coord))
# draw bbox and overlay mask into img and save it
np_mask = np.zeros_like(img).astype(np.uint8)
np_mask[:, :, 0] = np.where(masks[0], 255, 0)
mask = Image.fromarray(np_mask)
mask_img = Image.fromarray(overlay_mask(img, np_mask))
draw = ImageDraw.Draw(mask_img)
draw.rectangle(coord, outline=(0, 0, 255), width=1)
mask_img.save('./test.png')
