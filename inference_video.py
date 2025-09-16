import argparse
import os
import sys

import cv2
import imageio
import decord
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig
from model.segment_anything.utils.transforms import ResizeLongestSide



def parse_args(args):
    parser = argparse.ArgumentParser(description="EVF infer")
    parser.add_argument("--version", required=True)
    parser.add_argument("--vis_save_path", default="./infer", type=str)
    parser.add_argument(
        "--precision",
        default="fp16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=224, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)

    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--model_type", default="ori", choices=["ori", "effi", "sam2"])
    parser.add_argument("--video_path", type=str, default="assets/horse.mp4")
    parser.add_argument("--prompt", type=str, default="horse")
    
    return parser.parse_args(args)

def get_video_frames(video_path):
    video = decord.VideoReader(video_path)
    frames = video.get_batch(range(len(video)))
    frames = frames.asnumpy()
    return frames

def sam_preprocess(
    x: np.ndarray,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
    model_type="ori") -> torch.Tensor:
    '''
    preprocess of Segment Anything Model, including scaling, normalization and padding.  
    preprocess differs between SAM and Effi-SAM, where Effi-SAM use no padding.
    input: ndarray
    output: torch.Tensor
    '''
    assert img_size==1024, \
        "both SAM and Effi-SAM receive images of size 1024^2, don't change this setting unless you're sure that your employed model works well with another size."
    
    # Normalize colors
    if model_type=="ori":
        x = ResizeLongestSide(img_size).apply_image(x)
        h, w = resize_shape = x.shape[:2]
        x = torch.from_numpy(x).permute(2,0,1).contiguous()
        x = (x - pixel_mean) / pixel_std
        # Pad
        padh = img_size - h
        padw = img_size - w
        x = F.pad(x, (0, padw, 0, padh))
    else:
        x = torch.from_numpy(x).permute(2,0,1).contiguous()
        x = F.interpolate(x.unsqueeze(0), (img_size, img_size), mode="bilinear", align_corners=False).squeeze(0)
        x = (x - pixel_mean) / pixel_std
        resize_shape = None
    
    return x, resize_shape

def beit3_preprocess(x: np.ndarray, img_size=224) -> torch.Tensor:
    '''
    preprocess for BEIT-3 model.
    input: ndarray
    output: torch.Tensor
    '''
    beit_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC, antialias=None), 
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return beit_preprocess(x)

def init_models(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        padding_side="right",
        use_fast=False,
    )

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    if args.model_type=="ori":
        from model.evf_sam import EvfSamModel
        model = EvfSamModel.from_pretrained(
            args.version, low_cpu_mem_usage=True, **kwargs
        )
    elif args.model_type=="effi":
        from model.evf_effisam import EvfEffiSamModel
        model = EvfEffiSamModel.from_pretrained(
            args.version, low_cpu_mem_usage=True, **kwargs
        )
    elif args.model_type=="sam2":
        from model.evf_sam2 import EvfSam2Model
        model = EvfSam2Model.from_pretrained(
            args.version, low_cpu_mem_usage=True, **kwargs
        )

    if (not args.load_in_4bit) and (not args.load_in_8bit):
        model = model.cuda()
    model.eval()

    return tokenizer, model

def main(args):
    args = parse_args(args)
    # use float16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # clarify IO
    video_path = args.video_path
    if not os.path.exists(video_path):
        print("File not found in {}".format(video_path))
        exit()
    prompt = args.prompt

    # Don't create directory here since vis_save_path should be a file path, not a directory
    # The directory will be created later when we know the actual file path

    # initialize model and tokenizer
    tokenizer, model = init_models(args)

    # preprocess
    frames = get_video_frames(args.video_path)
    original_size_list = [frames[0].shape[:2]]
    frames_list = []

    for i, image_np in tqdm(enumerate(frames), total=len(frames)):
        image_beit = beit3_preprocess(image_np, args.image_size).to(dtype=model.dtype, device=model.device)

        image_sam, resize_shape = sam_preprocess(image_np, model_type=args.model_type)
        image_sam = image_sam.to(dtype=model.dtype, device=model.device)

        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device=model.device)

        # infer
        pred_mask = model.inference(
            image_sam.unsqueeze(0),
            image_beit.unsqueeze(0),
            input_ids,
            resize_list=[resize_shape],
            original_size_list=original_size_list,
        )
        pred_mask = pred_mask.detach().cpu().numpy()[0]
        pred_mask = pred_mask > 0
        pred_mask = (pred_mask * 255).astype(np.uint8)
        pred_mask = np.stack([pred_mask, pred_mask, pred_mask], axis=-1)
        frames_list.append(pred_mask)
        # mask_path = os.path.join(args.vis_save_path, f"{i:03d}.png")
        # cv2.imwrite(mask_path, pred_mask)
    
    # save video
    dir_save = os.path.dirname(args.vis_save_path)
    os.makedirs(dir_save, exist_ok=True)
    imageio.mimsave(args.vis_save_path, frames_list, fps=8, macro_block_size=1)

if __name__ == "__main__":
    main(sys.argv[1:])