import torch
from typing import Tuple, List

from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.inference import preprocess_caption

def predict_batch(
        model,
        images: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda"
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]]]:
    '''
    return: 
        bboxes_batch: list of tensors of shape (n, 4)
        predicts_batch: list of tensors of shape (n,)
        phrases_batch: list of list of strings of shape (n,)
        n is the number of boxes in one image
    '''
    caption = preprocess_caption(caption=caption)
    model = model.to(device)
    image = images.to(device)
    with torch.no_grad():
        outputs = model(image, captions=[caption for _ in range(len(images))]) # <------- I use the same caption for all the images for my use-case
    prediction_logits = outputs["pred_logits"].cpu().sigmoid()  # prediction_logits.shape = (num_batch, nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()  # prediction_boxes.shape = (num_batch, nq, 4)

    # import ipdb; ipdb.set_trace()
    mask = prediction_logits.max(dim=2)[0] > box_threshold # mask: torch.Size([num_batch, 256])
    
    bboxes_batch = []
    predicts_batch = []
    phrases_batch = [] # list of lists
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    for i in range(prediction_logits.shape[0]):
        logits = prediction_logits[i][mask[i]]  # logits.shape = (n, 256)
        phrases = [
                    get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
                    for logit # logit is a tensor of shape (256,) torch.Size([256])
                    in logits # torch.Size([7, 256])
                  ]
        boxes = prediction_boxes[i][mask[i]]  # boxes.shape = (n, 4)
        phrases_batch.append(phrases)
        bboxes_batch.append(boxes)
        predicts_batch.append(logits.max(dim=1)[0])
    
    return bboxes_batch, predicts_batch, phrases_batch