import os
os.environ['ATTN_BACKEND'] = 'xformers'

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from trellis import datasets, models
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from vq_ema import VectorQuantizerEMA
from torch.utils.data import DataLoader
from trellis.utils.data_utils import recursive_to_device
import numpy as np
import torch
from trellis.modules.sparse import SparseTensor
from trellis.renderers import GaussianRenderer
from easydict import EasyDict as edict
from typing import *
from trellis.utils.loss_utils import l1_loss, l2_loss, ssim, lpips

####################### params #######################

config = json.load(open("configs/vae/slat_vae_enc_dec_gs_swin8_B_64l8_fp16.json", 'r'))
data_dir = "datasets/ObjaSubset"
output_dir = "outputs/new_test"

path = "JeffreyXiang/TRELLIS-image-large"
enc_pretrained = f"{path}/ckpts/slat_enc_swin8_B_64l8_fp16"
dec_pretrained = f"{path}/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16"

codebook_size = 256
codebook_dim = 8
batch_size = 3
num_workers = 1
loss_type = "l1"
lambda_ssim = 0.2
lambda_lpips = 0.2
lambda_kl = 1e-06
regularizations = {
    "lambda_vol": 10000.0,
    "lambda_opacity": 0.001
}

####################### load models #######################

def load_model_state_dict(model, path: str):
    """
    Load a state dict into the given model from a checkpoint.

    Args:
        model: The model instance to load the state dict into.
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
    """
    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        model_file = f"{path}.safetensors"
    else:
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    model.load_state_dict(load_file(model_file))

model_dict = {
        name: getattr(models, model['name'])(**model['args']).cuda()
        for name, model in config['models'].items()
}

load_model_state_dict(model_dict['encoder'], enc_pretrained)
load_model_state_dict(model_dict['decoder'], dec_pretrained)

model_dict['VQ'] = VectorQuantizerEMA(
    codebook_size,
    codebook_dim,
    commitment_cost=0.25,
    decay=0.99,
    epsilon=1e-5
).cuda()

####################### set model to eval/train mode #######################

model_dict['encoder'].eval()
model_dict['decoder'].eval()
model_dict['VQ'].train()

####################### load data #######################

dataset = getattr(datasets, config['dataset']['name'])(data_dir, **config['dataset']['args'])
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers, # int(np.ceil(os.cpu_count() / torch.cuda.device_count())),
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
    collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
)

test_feats = torch.randn(1, 8)
test_coords = torch.randint(0, 10, (1, 4)).int()
test_sparse = SparseTensor(feats=test_feats, coords=test_coords)

####################### losses #######################

rendering_options = {"near" : 0.8,
                    "far" : 1.6,
                    "bg_color" : 'random'}
renderer = GaussianRenderer(rendering_options)
renderer.pipe.kernel_size = model_dict['decoder'].rep_config['2d_filter_kernel_size']

def _get_regularization_loss(self, reps: List[Gaussian]) -> Tuple[torch.Tensor, Dict]:
        loss = 0.0
        terms = {}
        if 'lambda_vol' in regularizations:
            scales = torch.cat([g.get_scaling for g in reps], dim=0)   # [N x 3]
            volume = torch.prod(scales, dim=1)  # [N]
            terms[f'reg_vol'] = volume.mean()
            loss = loss + regularizations['lambda_vol'] * terms[f'reg_vol']
        if 'lambda_opacity' in regularizations:
            opacity = torch.cat([g.get_opacity for g in reps], dim=0)
            terms[f'reg_opacity'] = (opacity - 1).pow(2).mean()
            loss = loss + regularizations['lambda_opacity'] * terms[f'reg_opacity']
        return loss, terms

def training_losses(
        feats: SparseTensor,
        image: torch.Tensor,
        alpha: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        return_aux: bool = False,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses.

        Args:
            feats: The [N x * x C] sparse tensor of features.
            image: The [N x 3 x H x W] tensor of images.
            alpha: The [N x H x W] tensor of alpha channels.
            extrinsics: The [N x 4 x 4] tensor of extrinsics.
            intrinsics: The [N x 3 x 3] tensor of intrinsics.
            return_aux: Whether to return auxiliary information.

        Returns:
            a dict with the key "loss" containing a scalar tensor.
            may also contain other keys for different terms.
        """
        z, mean, logvar = model_dict['encoder'](feats, sample_posterior=True, return_raw=True)
        reps = model_dict['decoder'](z)
        renderer.rendering_options.resolution = image.shape[-1]
        render_results = _render_batch(reps, extrinsics, intrinsics)     
        
        terms = edict(loss = 0.0, rec = 0.0)
        
        rec_image = render_results['color']
        gt_image = image * alpha[:, None] + (1 - alpha[:, None]) * render_results['bg_color'][..., None, None]
                
        if loss_type == 'l1':
            terms["l1"] = l1_loss(rec_image, gt_image)
            terms["rec"] = terms["rec"] + terms["l1"]
        elif loss_type == 'l2':
            terms["l2"] = l2_loss(rec_image, gt_image)
            terms["rec"] = terms["rec"] + terms["l2"]
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")
        if lambda_ssim > 0:
            terms["ssim"] = 1 - ssim(rec_image, gt_image)
            terms["rec"] = terms["rec"] + lambda_ssim * terms["ssim"]
        if lambda_lpips > 0:
            terms["lpips"] = lpips(rec_image, gt_image)
            terms["rec"] = terms["rec"] + lambda_lpips * terms["lpips"]
        
        terms["loss"] = terms["loss"] + terms["rec"]

        terms["kl"] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        terms["loss"] = terms["loss"] + lambda_kl * terms["kl"]
        
        reg_loss, reg_terms = _get_regularization_loss(reps)
        terms.update(reg_terms)
        terms["loss"] = terms["loss"] + reg_loss
        
        status = _get_status(z, reps)
        
        if return_aux:
            return terms, status, {'rec_image': rec_image, 'gt_image': gt_image}       
        return terms, status


######################################################

for batch in dataloader:
    batch = recursive_to_device(batch, torch.device('cuda'), non_blocking=True)
    # print(batch['feats'].shape)
    break
