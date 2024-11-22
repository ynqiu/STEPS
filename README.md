# STEPS: Sequential probability Tensor Estimation for text-to-tmage hard Prompt Search

STEPS is an efficient framework for optimizing text prompts in text-to-image (T2I) generation through sequential probability tensor decomposition.


## Installation

### Requirements
```
diffusers==0.11.1
ftfy==6.3.1
horovod==0.28.1
huggingface_hub==0.25.2
jax==0.4.34
numpy==2.1.3
optax==0.2.4
Pillow==11.0.0
regex==2024.9.11
Requests==2.32.3
sentence_transformers==2.2.2
timm==1.0.11
torch==1.13.0
torchvision==0.14.0
tqdm==4.66.5
transformers==4.23.1
```


## Key Arguments

STEPS provides several key parameters for optimization:

- `alg`: The running algorithm
- `prompt_len`: Length of the prompt sequence
- `iter`: Number of optimization iterations
- `rank`: Rank of tensor train decomposition
- `top_n`: Number of candidates to reduce the sequentially increasing the memory
- `sample_bs`: The maximum sampling size
- `dataset_name`: The dataset to run the algorithm



## Usage

1. Install dependencies:
```
pip install -r requirements.txt
```
2. Prepare your dataset in `data/`.

3. Configure parameters:

```
python run_STEPS.py \
--alg td \
--prompt_len 10 \
--iter 100 \
--rank 10 \
--topk 64
--top_n 8 \
--sample_bs 1000 \
--dataset_name coco \
```

4. Run optimization

For detailed examples, please refer to the code documentation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank the contributors and maintainers of the following projects that made STEPS possible:
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [Diffusers](https://github.com/huggingface/diffusers)
- [PyTorch](https://github.com/pytorch/pytorch)
- [PROTES](https://github.com/anabatsh/PROTES)