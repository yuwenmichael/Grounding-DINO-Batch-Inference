# GroundingDINO Infer in Batches

# Table of Contents
- [GroundingDINO Infer in Batches](#groundingdino-infer-in-batches)
- [Table of Contents](#table-of-contents)
  - [Inspirations](#inspirations)
  - [Instructions](#instructions)
    - [1. Prepare the images](#1-prepare-the-images)
    - [2. Create file path list](#2-create-file-path-list)
    - [3. Run the inference](#3-run-the-inference)
  - [Attentions](#attentions)
## Inspirations
By following this github [post](https://github.com/IDEA-Research/GroundingDINO/issues/102#issuecomment-1558728065) provided by [ashrielbrian](https://github.com/ashrielbrian), I managed to get the inference to work in batches. Modifications need to be made as the original post is only testing the latency of the model. The modifications are made in the `batch_utlities.py` file.


Here is a copy of what he mentioned in the github post:

I (He) managed to get a batch script to work in a somewhat hacky way:

Stack the images into a batch:
```
images = torch.stack([load_image(img)[1] for img in img_paths])
boxes, logits, phrases = predict_batch(
        model=model,
        images=images,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
```
You'll need to update the `load_image` func to not use the `RandomResize`. Inside `datasets/transforms.py`, add this class:
```
class Resize(object):
    def __init__(self, size):
        assert isinstance(size, (list, tuple))
        self.size = size

    def __call__(self, img, target=None):
        return resize(img, target, self.size)
```

Inside `load_image` in `inference.py`, I hardcoded the resize to ensure every image in the batch is of the same size. This is hacky and probably (definitely) results in poorer performance.
```
transform = T.Compose(
        [
            # T.RandomResize([800], max_size=1333),
            # Added T.Resize to fix the resized image during batch inference
            T.Resize((800, 1200)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
```
Adapting the existing `predict` function:

```
def predict_batch(
        model,
        images: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    caption = preprocess_caption(caption=caption)

    model = model.to(device)
    image = images.to(device)

    print(f"Image shape: {image.shape}") # Image shape: torch.Size([num_batch, 3, 800, 1200])
    with torch.no_grad():
        outputs = model(image, captions=[caption for _ in range(len(images))]) # <------- I use the same caption for all the images for my use-case

    print(f'{outputs["pred_logits"].shape}') # torch.Size([num_batch, 900, 256]) 
    print(f'{outputs["pred_boxes"].shape}') # torch.Size([num_batch, 900, 4])
    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    phrases = [
        get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
        for logit
        in logits
    ]

    return boxes, logits.max(dim=1)[0], phrases
```
This gave me (him) a roughly 18% improvement in latency over single image inference of a batch of 16 images.


## Instructions
To begin with, since this project is based on [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), please follow the instructions in the original repo to set up the environment and download the pretrained model and put the weigth file in the `./weights` folder. The weight I use is 'groundingdino_swint_ogc.pth'. Another thing to mention is that the prompt I use is 'object' with a batch size of 2, you can modify the parameters as you need in the `inference_gdino.py` file. Moreover, I am using cpu, please modify the code in `inference_gdino.py` to use gpu if you want to.

### 1. Prepare the images
Put the images you want to infer in the `./images` folder. The images can be in any format, but the code only supports `.jpg` for now.

### 2. Create file path list
Create a file path list By running ``` python create_img_path_list.py```. This will create a file path list for the images you want to infer. The file path list should be in the `./images` folder and named `img_paths.txt`. Each line in the file should be the path to the image. For example:
```
10072701692931/38257c9f278852af.jpg
10072701692931/c6f407dd59d06a81.jpg
71230141340/2701e43b25f70394.jpg
71230141340/0f70e523ef92d29a.jpg
```
### 3. Run the inference
```
python inference_gdino.py
```
The results will be saved in the `./crop_images` folder, where the name of the folder is the sku (aka product ID) of a product.

## Attentions
The result is already presented in the `crop_images` folder in this repo. If you want to run the program again, please make sure you delete the `crop_images` folder first. Otherwise, the program will not run properly.