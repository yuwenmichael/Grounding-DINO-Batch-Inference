from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
from batch_utlities import predict_batch
import cv2
from tqdm import tqdm
import os
from torchvision.ops import box_convert
import torch
from PIL import Image

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", type=str, default="images",
                    help="root dir for images")
parser.add_argument("--image_list_file", type=str, default='image_paths.txt',
                    help="paths of image list")
parser.add_argument("--visual_dir", type=str, default=None,
                    help="folder to visual")
parser.add_argument("--save_crop_dir", type=str, default="crop_images",
                    help="folder to visual")
parser.add_argument("--text_prompt", type=str, default='object',
                    help="detection prompt")
args = parser.parse_args()

CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "weights/groundingdino_swint_ogc.pth"
DEVICE = "cpu"
# DEVICE = "cuda"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25
imgType_list = {'.jpg', '.bmp', '.png', '.jpeg', '.JPG', '.JPEG', '.tif'}

model = load_model(CONFIG_PATH, CHECKPOINT_PATH, DEVICE)

print('loading image list file from: ', args.image_list_file)

# one image path per line: 
# example: 10072701692931/38257c9f278852af.jpg
image_list = [_.strip().split(',')[0] for _ in open(args.image_list_file, 'r').readlines()]

final_input_paths, final_save_paths = [], []
for image_name in image_list:
    # if os.path.splitext(image_name)[-1] not in imgType_list:
    #     image_name = f'{image_name}.jpg'
    image_path = image_name if args.image_dir is None else os.path.join(args.image_dir, image_name)
    if args.image_dir is None:
        crop_path = os.path.join(args.save_crop_dir, image_path.split('/')[1])
    else:
        crop_path = image_path.replace(args.image_dir, args.save_crop_dir)
    if os.path.exists(crop_path): continue
    final_input_paths.append(image_path)
    final_save_paths.append(crop_path)
    
print(
    f'total images:{len(image_list)}, need detect: {len(final_input_paths)}, skip images: {len(image_list) - len(final_input_paths)}')


def batch_load_images(image_paths_lst, final_save_paths, batch_size):
    num_batches = (len(image_paths_lst) + batch_size - 1) // batch_size

    batches_image_path = []
    batches_save_path = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(image_paths_lst))
        batch_paths = image_paths_lst[start_idx:end_idx]
        batch_save_path = final_save_paths[start_idx:end_idx]
        
        batch_images_path = [path for path in batch_paths]
        batch_save_path = [path for path in batch_save_path]
        
        batches_image_path.append(batch_images_path)
        batches_save_path.append(batch_save_path)
    
    return num_batches, batches_save_path, batches_image_path

batch_size = 2
num_batches, batches_save_path, batches_image_path = batch_load_images(final_input_paths, final_save_paths, batch_size)


with tqdm(total=num_batches) as _tqdm:
    _tqdm.set_description(f'detect: ')
    
    for batch_idx in range(num_batches):
        
        img_paths = batches_image_path[batch_idx]
        images = torch.stack([load_image(img)[1] for img in img_paths])
        boxes, logits, phrases = predict_batch(
            model=model,
            images=images,
            caption=args.text_prompt,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device=DEVICE
        )
        # import ipdb; ipdb.set_trace()
        
        for i in range(len(img_paths)):
            # intermidiate = Image.open(img_paths[i]).convert("RGB")
            image_source, _ = load_image(img_paths[i])
            if args.save_crop_dir is not None:
                # TODO: save crop image to dir
                h, w, _ = image_source.shape
                # import ipdb; ipdb.set_trace()
                boxes_ = boxes[i] * torch.Tensor([w, h, w, h])
                xyxys = box_convert(boxes=boxes_, in_fmt="cxcywh", out_fmt="xyxy").numpy().tolist()
                crop_path = batches_save_path[batch_idx][i]
                os.makedirs(os.path.dirname(crop_path), exist_ok=True)
                if len(xyxys) > 0:
                    # find the max area
                    max_area_idx = 0
                    max_area = 0
                    for idx, xyxy in enumerate(xyxys):
                        x1, y1, x2, y2 = [int(_) for _ in xyxy]
                        area = (x2 - x1) * (y2 - y1)
                        max_area = max(max_area, area)
                        max_area_idx = idx if area == max_area else max_area_idx
                    x1, y1, x2, y2 = [int(_) for _ in xyxys[max_area_idx]]
                else:
                    x1, y1, x2, y2 = 0, 0, w, h
                try:
                    crop_image = image_source[y1:y2, x1:x2, :]
                    # import ipdb
                    # ipdb.set_trace()
                    # print('crop image into ', crop_path)
                    # cv2.imwrite(crop_path, crop_image)
                    Image.fromarray(crop_image).save(crop_path)
                except:
                    Image.fromarray(image_source).save(crop_path)
            # import ipdb; ipdb.set_trace()
        _tqdm.update(1)