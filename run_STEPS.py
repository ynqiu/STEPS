import os
import json
import logging
from PIL import Image
from typing import List
from src.utils import *
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
import argparse
from datetime import datetime
import torch
import open_clip
import gc   
import time
import numpy as np
import random

def read_images_from_folder(folder_path: str, num_images: int = None) -> List[Image.Image]:
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith((".png", ".jpg", ".jpeg"))]
    if num_images is not None:
        image_paths = image_paths[:num_images]
    return [Image.open(path) for path in image_paths]

def generate_images(prompt: str, num_images: int, pipe: StableDiffusionPipeline, image_length: int, guidance_scale: float, num_inference_steps: int, output_dir: str, image_name: str):
    images = pipe(
        prompt,
        num_images_per_prompt=num_images,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=image_length,
        width=image_length,
    ).images

    for i, image in enumerate(images, start=1):
        image_path = os.path.join(output_dir, f"{image_name}-{i}.png")
        image.save(image_path)

def measure_similarity(orig_image: Image.Image, gen_image: Image.Image, ref_model, ref_clip_preprocess, device: str) -> float:
    with torch.no_grad():
        ori_feat = ref_model.encode_image(ref_clip_preprocess(orig_image).unsqueeze(0).to(device))
        gen_feat = ref_model.encode_image(ref_clip_preprocess(gen_image).unsqueeze(0).to(device))

        ori_feat = ori_feat / ori_feat.norm(dim=1, keepdim=True)
        gen_feat = gen_feat / gen_feat.norm(dim=1, keepdim=True)

        return (ori_feat @ gen_feat.t()).item()

def get_output_dir(args):
    """根据不同算法构建输出路径"""
    base_dir = os.path.join("results", args.dataset_name)
    
    if args.alg == "td":
        method_params = (
            f"{args.alg}_"
            f"len_{args.prompt_len}_"
            f"topk_{args.topk}_"
            f"bs_{args.sample_bs}_"
            f"topn_{args.top_n}_"
            f"r_{args.rank}_"
            f"iter_{args.iter}_"
            f"seed_{args.seed}"
        )
    else:
        method_params = f"{args.alg}_prompt_len_{args.prompt_len}"
    
    return os.path.join(base_dir, method_params)

def main(args):
    # Stage 1: Optimizing prompts
    logging.info("Stage 1: Optimizing prompts...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrain, device=device)
    
    orig_images = read_images_from_folder(os.path.join(args.dataset_path), args.num_images)
    prompts = {}
    
    output_dir = get_output_dir(args)
    print(f"\033[94m{output_dir}\033[0m")

    for i, orig_image in enumerate(orig_images, start=1):
        image_name = os.path.splitext(os.path.basename(orig_image.filename))[0]
        logging.info(f"Processing image {i}/{len(orig_images)}: {image_name}")
        
        learned_prompt, best_sim = optimize_prompt(model, preprocess, args, device, target_images=[orig_image])
        prompts[image_name] = {
            "prompt": learned_prompt,
            "best_sim": best_sim
        }
        logging.info(f"Learned prompt: {learned_prompt}, Best similarity: {best_sim:.4f}")

    # Save prompts
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "prompts.json"), "w") as f:
        json.dump(prompts, f, indent=2)

    # Release memory for the first stage
    del model, preprocess
    torch.cuda.empty_cache()
    gc.collect()
    logging.info("Stage 1 completed and memory cleared")

    # Stage 2: Generating images
    logging.info("Stage 2: Generating images...")
    model_id = "stabilityai/stable-diffusion-2-base"
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, 
                                                  torch_dtype=torch.float16, revision="fp16").to(device)
    image_length = 512
    num_images = 5
    guidance_scale = 9
    num_inference_steps = 25

    for image_name, prompt_data in prompts.items():
        generate_images(prompt_data["prompt"], num_images, pipe, image_length, guidance_scale, 
                       num_inference_steps, output_dir, image_name)
        logging.info(f"Generated images for {image_name}")

    # Release memory for the second stage
    del pipe, scheduler
    torch.cuda.empty_cache()
    gc.collect()
    logging.info("Stage 2 completed and memory cleared")

    # Stage 3: Calculating similarities
    logging.info("Stage 3: Calculating similarities...")
    ref_model, _, ref_preprocess = open_clip.create_model_and_transforms(args.ref_clip_model, 
                                                                        pretrained=args.ref_clip_pretrain, 
                                                                        device=device)

    results = {}
    for i, orig_image in enumerate(orig_images, start=1):
        image_name = os.path.splitext(os.path.basename(orig_image.filename))[0]
        
        similarities = []
        for j in range(1, num_images + 1):
            gen_image_path = os.path.join(output_dir, f"{image_name}-{j}.png")
            gen_image = Image.open(gen_image_path)
            similarity = measure_similarity(orig_image, gen_image, ref_model, ref_preprocess, device)
            similarities.append(similarity)

        results[os.path.basename(orig_image.filename)] = {
            "prompt": prompts[image_name]["prompt"],
            "best_sim": prompts[image_name]["best_sim"],
            "similarities": similarities,
            "avg_similarity": sum(similarities) / len(similarities),
            "method": args.alg,
            "method_params": vars(args),
        }

    # Calculate and save results
    avg_similarities = [result["avg_similarity"] for result in results.values()]
    overall_avg_similarity = sum(avg_similarities) / len(avg_similarities)
    results_filename = f"{args.alg}_avg{overall_avg_similarity:.4f}_results.json"

    with open(os.path.join(output_dir, results_filename), "w") as f:
        json.dump(results, f, indent=2)

    logging.info(f"Overall average similarity: {overall_avg_similarity:.4f}")   

    # Release memory for the third stage
    del ref_model, ref_preprocess
    torch.cuda.empty_cache()
    gc.collect()
    logging.info("Stage 3 completed and memory cleared")

    # Record all parameters to the log
    logging.info("All parameters:")
    for arg_name, arg_value in vars(args).items():
        logging.info(f"{arg_name}: {arg_value}")

def load_config(method):
    """Load configuration from the unified config file"""
    try:
        with open("config.json", 'r') as f:
            configs = json.load(f)
        return configs
    except FileNotFoundError:
        print("Config file not found, using default parameters")
        return None

def setup_logging(args):
    """Set up logging configuration"""
    os.makedirs("logs", exist_ok=True)
    
    # Build basic parameters part
    base_params = (
        f"alg_{args.alg}_"
        f"dataset_{args.dataset_name}_"
        f"len_{args.prompt_len}_"
    )
    
    # Add specific parameters according to different algorithms
    if args.alg == "td":
        method_params = (
            f"topk_{args.topk}_"
            f"bs_{args.sample_bs}_"
            f"topn_{args.top_n}_"
            f"r_{args.rank}_"
            f"iter_{args.iter}_"
            f"seed_{args.seed}_"
        )
    else:
        method_params = ""
    
    # Combine file name
    filename = 'logs/' + base_params + method_params + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log'
    
    # Set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(filename)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    for handler in [file_handler, console_handler]:
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
    
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.info(f"Started new run with args: {args}")
    logging.info(f"Log file created at: {filename}")
    
    return logger

def setup_parser():
    parser = argparse.ArgumentParser(description='Run diffusion model with CLIP')
    
    # Basic parameters
    parser.add_argument('--alg', type=str, default='td', help='Algorithm to run')
    parser.add_argument('--dataset_name', type=str, default='coco', help='Dataset name')
    parser.add_argument('--num_images', type=int, default=None, help='Number of images to process (None for all images)')
    parser.add_argument('--prompt_len', type=int, default=10, help='Prompt length')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')

    # CLIP model parameters
    parser.add_argument('--clip_model', type=str, default='ViT-H-14')
    parser.add_argument('--clip_pretrain', type=str, default='laion2b_s32b_b79k')
    parser.add_argument('--ref_clip_model', type=str, default='ViT-big-14')
    parser.add_argument('--ref_clip_pretrain', type=str, 
                       default='../../clipmodel/CLIP-ViT-bigG/open_clip_pytorch_model.bin')
    
    # STEPS method specific parameters
    parser.add_argument('--iter', type=int, default=200, help='Number of iterations')
    parser.add_argument('--sample_bs', type=int, default=1000, help='Sample batch size')
    parser.add_argument('--topk', type=int, default=64, help='Top k')
    parser.add_argument('--rank', type=int, default=10, help='rank r for STEPS')
    parser.add_argument('--top_n', type=int, default=8, help='Top n for STEPS')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--k_top', type=int, default=10)
    parser.add_argument('--prompt_bs', type=int, default=1)
    parser.add_argument('--loss_weight', type=float, default=1.0)
    parser.add_argument('--remark', type=str, default='', help='Remark')
    parser.add_argument('--update_topn', type=bool, default=False)


    
    return parser

def set_random_seed(seed=None):
    """
    Set the seed for all random number generators
    """
    if seed is None:
        seed = int(time.time() * 1000) % 1000000
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # If using CUDA, ensure that the results are deterministic
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logging.info(f"Using random seed: {seed}")

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.seed)

    # Set dataset path
    dataset_paths = {
        'coco': 'data/coco100',
        'flick30k': 'data/flick30k100',
        'laion': 'data/laion100',
        'celeba': 'data/celeba100',
        'textvqa': 'data/textvqa100',
        'vizwiz': 'data/vizwiz100',
    }

    if args.dataset_name in dataset_paths:
        args.dataset_path = dataset_paths[args.dataset_name]
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")
    
    logger = setup_logging(args)
    main(args)
