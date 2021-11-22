# Perception Prioritized Training of Diffusion Models

This implementation is heavily based on [guided-diffusion](https://github.com/openai/guided-diffusion).

## Preparation
Our code is tested on PyTorch 1.7.

Set PYTHONPATH variable to point to the root of the repository.
```
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

We provide models trained with our P2 weighting on various datasets.
Here are the download links for each model checkpoint:

* FFHQ: [ffhq.pt](https://drive.google.com/file/d/1TFMVZATTzbE253J6KyFXq1uOJNODL-oP/view?usp=sharing)
* CelebA-HQ: [celeb.pt](https://drive.google.com/file/d/1I8ltTRwJrSNWQlJ8apVYrVjPBj75JlmA/view?usp=sharing)
* MetFaces: [metface_drop.pt](https://drive.google.com/file/d/13cjQ4TX2665t5uUxYcA0jzCMojjuVDD7/view?usp=sharing)
* CUB: [cub_drop.pt](https://drive.google.com/file/d/1Hv7aSdVIprqXBAtBcePi64_u4U7aw7gZ/view?usp=sharing)
* Flowers: [flowers_drop.pt](https://drive.google.com/file/d/1aXeuJbzZaxPU7GtK10vW1YQkh9BUIpUT/view?usp=sharing)

## Sampling
Specify model confiurations and save directory.
Run the below script.

```
python scripts/image_sample.py --attention_resolutions 16 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 1 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --timestep_respacing 250 --use_ddim False --model_path YOUR_MODEL_PATH --save_dir YOUR_SAVE_PATH
```

To sample for 25 timesteps with DDIM, replace `--timestep_respacing 250` to `--timestep_respacing ddim25`, and replace `--use_ddim False` to `--use_ddim True`.

## Training
Specify model confiurations, dataset directory, and log directory to save your logs and models.
Running below script creates YOUR_LOG_PATH automatically.

```
python scripts/image_train.py --data_dir YOUR_DATA_PATH --attention_resolutions 16 --class_cond False --diffusion_steps 1000 --dropout 0.0 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 1 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --lr 2e-5 --batch_size 8 --rescale_learned_sigmas True --log_dir YOUR_LOG_PATH
```
