#!/usr/bin/env python
# coding: utf-8

# In[33]:


import torch
import gc
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def clear_gpu():
    torch.cuda.empty_cache()
    gc.collect()

clear_gpu()

# ### Attempt \#2 at figuring out how to integrate GLIDE

# #### *STEP 1:* Set up GLIDE Model Training

# In[34]:


from PIL import Image
from IPython.display import display
from torchvision.utils import save_image

import torch as th
from torch.optim import AdamW
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)


# In[35]:


# --- 1. CONFIGURATION & HYPERPARAMETERS ---
has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')
BATCH_SIZE = 4

LEARNING_RATE = 1e-5
EPOCHS = 40
IMAGE_SIZE = 64
UPSAMPLE_FACTOR = 4 # 64 -> 256


# In[36]:


# --- 2. INITIALIZE MODELS WITH PRE-TRAINED WEIGHTS ---
# Base Model Configuration
base_options = model_and_diffusion_defaults()
base_options['use_fp16'] = has_cuda
base_options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
base_model, base_diffusion = create_model_and_diffusion(**base_options)
if has_cuda:
    base_model.convert_to_fp16()
base_model.to(device)
base_model.load_state_dict(load_checkpoint('base', device))
print('total base parameters', sum(x.numel() for x in base_model.parameters()))

# Upsampler Configuration
up_options = model_and_diffusion_defaults_upsampler()
up_options['use_fp16'] = has_cuda
up_options['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
up_model, up_diffusion = create_model_and_diffusion(**up_options)
if has_cuda:
    up_model.convert_to_fp16()
up_model.to(device)
up_model.load_state_dict(load_checkpoint('upsample', device))
print('total upsampler parameters', sum(x.numel() for x in up_model.parameters()))


# In[37]:


# --- 3. OPTIMIZERS ---
# Using AdamW for weight decay support
optimizer_base = AdamW(base_model.parameters(), lr=LEARNING_RATE)
optimizer_up = AdamW(up_model.parameters(), lr=LEARNING_RATE)


# In[38]:


import torch as th
import torch.nn.functional as F

def run_fine_tuning(dataloader):
    # 1. Force models to float32 to ensure backward pass compatibility
    # Autocast will handle the speedup/FP16 forward pass
    base_model.float()
    up_model.float()

    base_model.train()
    up_model.train()

    label_names = dataloader.dataset.features['label'].names
    max_text_len = base_options.get("text_ctx", 64)

    for epoch in range(EPOCHS):
        for i, (images, labels) in enumerate(dataloader):
            B = images.shape[0]

            # Keep images in float32; autocast will downcast them internally
            images = images.to(device).float() 
            base_images = F.interpolate(images, (IMAGE_SIZE, IMAGE_SIZE))

            # --- TOKENIZATION ---
            captions = [f"a photo of {label_names[int(l)]}" for l in labels]
            tokens_list, mask_list = [], []
            for caption in captions:
                raw_tokens = base_model.tokenizer.encode(caption)
                tokens = raw_tokens[:max_text_len]
                mask = [True] * len(tokens)
                while len(tokens) < max_text_len:
                    tokens.append(0)
                    mask.append(False)
                tokens_list.append(tokens)
                mask_list.append(mask)
            token_tensor = th.tensor(tokens_list, dtype=th.long, device=device)
            mask_tensor = th.tensor(mask_list, dtype=th.bool, device=device)
            model_kwargs = {"tokens": token_tensor, "mask": mask_tensor}

            # --- TRAIN BASE MODEL ---
            optimizer_base.zero_grad(set_to_none=True)

            # Use the modern torch.amp.autocast for safer mixed precision
            with th.amp.autocast('cuda'):
                t_base = th.randint(0, base_diffusion.num_timesteps, (B,), device=device)
                noise_base = th.randn_like(base_images)
                x_t_base = base_diffusion.q_sample(base_images, t_base, noise=noise_base)

                model_output_base = base_model(x_t_base, t_base, **model_kwargs)
                eps_pred_base, _ = th.split(model_output_base, 3, dim=1)
                loss_base = F.mse_loss(eps_pred_base, noise_base)

            if not th.isnan(loss_base):
                # No .float() needed here because weights are now float32
                loss_base.backward()
                th.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
                optimizer_base.step()

            del model_output_base, eps_pred_base, x_t_base
            th.cuda.empty_cache()

            # --- TRAIN UPSAMPLER ---
            optimizer_up.zero_grad(set_to_none=True)

            with th.amp.autocast('cuda'):
                t_up = th.randint(0, up_diffusion.num_timesteps, (B,), device=device)
                noise_up = th.randn_like(images)
                x_t_up = up_diffusion.q_sample(images, t_up, noise=noise_up)

                up_kwargs = model_kwargs.copy()
                up_kwargs["low_res"] = base_images

                model_output_up = up_model(x_t_up, t_up, **up_kwargs)
                eps_pred_up, _ = th.split(model_output_up, 3, dim=1)
                loss_up = F.mse_loss(eps_pred_up, noise_up)

            del model_output_up, eps_pred_up, x_t_up
            th.cuda.empty_cache()

            if not th.isnan(loss_up):
                loss_up.backward()
                th.nn.utils.clip_grad_norm_(up_model.parameters(), 1.0)
                optimizer_up.step()

            if i % 50 == 0:
                print(f"[Epoch {epoch}] {(i / len(dataloader) * 100):.2f}% ({i}/{len(dataloader)} batches) | Step {i} | Base: {loss_base.item():.4f} | Up: {loss_up.item():.4f}")


# #### *STEP 2:* Executing Model Training for the Dataset
# 1. ***Define the Preprocessing for GLIDE:*** GLIDE expects images to be normalized to a range of [−1,1] and resized to the dimensions specified in your config (128 for base, 256 for high-res).
# 2. ***Create a "Collate" Function:*** The DataLoader needs to know how to stack your list of samples into a single batch of Tensors and a list of text captions.
# 3. ***Initialize the DataLoader***

# In[39]:


from huggingface_hub import notebook_login
notebook_login()

import sys
from datasets import load_dataset
train_ds = load_dataset("imagefolder", data_dir="/data/team01/ds340w/datasets/GLIDE_split")["train"]


# In[40]:


import torch as th
from torchvision import transforms
from torch.utils.data import DataLoader

# 128 for the base model, but we keep 256 for the upsampler target
IMAGE_SIZE_UP = 256 

# GLIDE standard normalization: maps [0, 1] to [-1, 1]
glide_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE_UP, IMAGE_SIZE_UP)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def transform_fn(examples):
    # Apply transforms to the 'image' column
    examples["pixel_values"] = [glide_transforms(image.convert("RGB")) for image in examples["image"]]
    return examples

# Apply the transform to your dataset
train_ds.set_transform(transform_fn)


# In[41]:


def collate_fn(batch):
    # 1. Stack the pixel values into a [B, 3, 256, 256] tensor
    pixel_values = th.stack([item["pixel_values"] for item in batch])

    # 2. Extract captions. 
    # If "label" is an integer, convert it to a string or look up its name.
    captions = []
    for item in batch:
        label = item["label"]
        # If your dataset has a feature 'label_names', use it here
        # Otherwise, just ensure it's a string
        captions.append(str(label)) 

    return pixel_values, captions


# In[42]:


train_dataloader = DataLoader(
    train_ds, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=collate_fn,
    drop_last=True
)

# print(train_dataloader.dataset.features)

# --- Start Training ---
# Now pass it into your function
run_fine_tuning(train_dataloader)


# In[ ]:


th.save(base_model.state_dict(), "/data/team01/ds340w/Enhancing-Skin-Disease-Classification/New_Enhanced_2/base_finetuned_v2.pth") # Base (128 x 128 model)
th.save(up_model.state_dict(), "/data/team01/ds340w/Enhancing-Skin-Disease-Classification/New_Enhanced_2/up_finetuned_v2.pth")     # Up (256 x 256 model)


# In[ ]:


# import torch as th

# # Define the paths
# base_path = "/data/team01/ds340w/Enhancing-Skin-Disease-Classification/New_Enhanced_2/base_finetuned_v1.pth"
# up_path = "/data/team01/ds340w/Enhancing-Skin-Disease-Classification/New_Enhanced_2/up_finetuned_v1.pth"

# # 1. Load the state dictionaries
# base_state_dict = th.load(base_path, map_location=device)
# up_state_dict = th.load(up_path, map_location=device)

# # 2. Load the weights into your existing model objects
# base_model.load_state_dict(base_state_dict)
# up_model.load_state_dict(up_state_dict)

# # 3. Set to evaluation mode
# # This is crucial! It disables dropout and batch norm updates.
# base_model.eval()
# up_model.eval()

# # 4. (Optional) If you are using half-precision/bfloat16 for inference
# base_model.to(th.bfloat16) 
# up_model.to(th.bfloat16)

# print("Finetuned Dermnet models loaded successfully.")


# #### *STEP 3:* Synthesize Model Given Parameters

# In[ ]:


def show_images(batch: th.Tensor):
    """ Display a batch of images inline. """
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    display(Image.fromarray(reshaped.numpy()))


# In[ ]:


#########################################
# Sample from the base model + Upsample #
#########################################

def sample_and_upsample(prompt, batch_size, guidance_scale, upsample_temp):
    # Create the text tokens to feed to the model.
    tokens = base_model.tokenizer.encode(prompt)
    tokens, mask = base_model.tokenizer.padded_tokens_and_mask(
        tokens, base_options['text_ctx']
    )

    # Create the classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2
    uncond_tokens, uncond_mask = base_model.tokenizer.padded_tokens_and_mask(
        [], base_options['text_ctx']
    )

    # Pack the tokens together into model kwargs.
    model_kwargs = dict(
        tokens=th.tensor(
            [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),
    )

    # Create a classifier-free guidance sampling function
    def model_fn(x_t, ts, **kwargs):
        # 1. Force everything inside this function to use consistent dtypes
        with th.amp.autocast('cuda'):
            half = x_t[: len(x_t) // 2]
            combined = th.cat([half, half], dim=0)

            # Ensure the combined tensor matches the model's primary weight type
            combined = combined.to(base_model.dtype)

            model_out = base_model(combined, ts, **kwargs)

            # 2. Extract outputs
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)

            # 3. Apply guidance
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = th.cat([half_eps, half_eps], dim=0)

            return th.cat([eps, rest], dim=1)

    # Sample from the base model.
    base_model.del_cache()
    samples = base_diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, base_options["image_size"], base_options["image_size"]),
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]
    base_model.del_cache()

    # show_images(samples)

    tokens = up_model.tokenizer.encode(prompt)
    tokens, mask = up_model.tokenizer.padded_tokens_and_mask(
        tokens, up_options['text_ctx']
    )

    # Create the model conditioning dict.
    model_kwargs = dict(
        # Low-res image to upsample.
        low_res=((samples+1)*127.5).round()/127.5 - 1,

        # Text tokens
        tokens=th.tensor(
            [tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            [mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),
    )

    # Sample from the base model.
    up_model.del_cache()
    up_shape = (batch_size, 3, up_options["image_size"], up_options["image_size"])
    with th.amp.autocast('cuda'):
        up_samples = up_diffusion.ddim_sample_loop(
            up_model,
            up_shape,
            noise=th.randn(up_shape, device=device) * upsample_temp,
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
    up_model.del_cache()

    return up_samples
    # show_images(up_samples)


# In[ ]:


names = train_dataloader.dataset.features['label'].names
for name in names:
    # Tune this parameter to control the sharpness of 256x256 images.
    # A value of 1.0 is sharper, but sometimes results in grainy artifacts.
    guidance_scale = 3.0

    sd_prompt = f"a photo of {name}"
    batch_size = 4

    # Generate 100x images/disease (could be 50x after all is said-and-done)
    num_batches = 25
    for i in range(num_batches):
        upsample_temp = 0.997
        up_samples = sample_and_upsample(sd_prompt, batch_size, guidance_scale, upsample_temp)

        output_dir = f"/data/team01/ds340w/datasets/GLIDE_synthesis/{name}"
        os.makedirs(output_dir, exist_ok=True)
        for j, image_tensor in enumerate(up_samples):
            # normalize=True shifts the images from [-1, 1] to [0, 1] for proper saving
            file_path = os.path.join(output_dir, f"sample_{i*batch_size+j}.png")
            save_image(image_tensor, file_path, normalize=True)

    print(f"Saved {batch_size*num_batches} images to {output_dir}")


# In[ ]:




