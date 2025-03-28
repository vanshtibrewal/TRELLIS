import os
os.environ['ATTN_BACKEND'] = 'xformers'

import torch
import numpy as np
import pandas as pd
import trellis.models as models
import trellis.modules.sparse as sp
from huggingface_hub import hf_hub_download
import json
import imageio
from trellis.utils import render_utils

VERBOSE = False

torch.set_grad_enabled(False)

def main():
    output_dir = "datasets/ObjaSubset"
    feat_model = "dinov2_vitl14_reg"
    path = "JeffreyXiang/TRELLIS-image-large"
    enc_pretrained = f"{path}/ckpts/slat_enc_swin8_B_64l8_fp16"

    config_file = hf_hub_download(path, "pipeline.json")
    with open(config_file, 'r') as f:
        args = json.load(f)['args']

    dec_mesh_pretrained = f"{path}/{args['models']['slat_decoder_mesh']}"
    dec_gs_pretrained = f"{path}/{args['models']['slat_decoder_gs']}"
    dec_rf_pretrained = f"{path}/{args['models']['slat_decoder_rf']}"

    encoder = models.from_pretrained(enc_pretrained).eval().cuda()
    dec_mesh = models.from_pretrained(dec_mesh_pretrained).eval().cuda()
    dec_gs = models.from_pretrained(dec_gs_pretrained).eval().cuda()
    dec_rf = models.from_pretrained(dec_rf_pretrained).eval().cuda()
    
    metadata = pd.read_csv(os.path.join(output_dir, 'metadata.csv'))
    metadata = metadata[metadata[f'feature_{feat_model}'] == True]
    sha256 = metadata['sha256'].values[0]
    localpath = metadata['local_path'].values[0]

    if VERBOSE:
        print("Local ground truth path: ", localpath)
    print("Original Input: ", metadata['file_identifier'].values[0])

    feats = np.load(os.path.join(output_dir, 'features', feat_model, f'{sha256}.npz'))
    if VERBOSE:
        print("Features loaded:")
        print(f"patchtokens: {feats['patchtokens'].shape}")
        print(f"indices: {feats['indices'].shape}")

    # Process through encoder
    feats = sp.SparseTensor(
        feats = torch.from_numpy(feats['patchtokens']).float(),
        coords = torch.cat([
            torch.zeros(feats['patchtokens'].shape[0], 1).int(),
            torch.from_numpy(feats['indices']).int(),
        ], dim=1),
    ).cuda()
    
    latent = encoder(feats, sample_posterior=False)
    assert torch.isfinite(latent.feats).all(), "Non-finite latent"

    if VERBOSE:
        print("Latent encoded:")
        print(f"latent.feats.shape: {latent.feats.shape}")
        print(f"latent.coords.shape: {latent.coords.shape}")

    mesh = dec_mesh(latent)
    gaussian = dec_gs(latent)
    radiance_field = dec_rf(latent)
    
    print("Reconstructed videos saving to sample_gs.mp4, sample_rf.mp4, sample_mesh.mp4")
    video = render_utils.render_video(gaussian[0])['color']
    imageio.mimsave("sample_gs.mp4", video, fps=30)
    video = render_utils.render_video(radiance_field[0])['color']
    imageio.mimsave("sample_rf.mp4", video, fps=30)
    video = render_utils.render_video(mesh[0])['normal']
    imageio.mimsave("sample_mesh.mp4", video, fps=30)

if __name__ == '__main__':
    main()
