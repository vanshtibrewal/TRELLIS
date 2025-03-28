import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['ATTN_BACKEND'] = 'xformers'

import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import trellis.models as models
import trellis.modules.sparse as sp
from vq_ema import VectorQuantizerEMA

DEBUG = False

def main():
    output_dir = "datasets/ObjaSubset"
    feat_model = "dinov2_vitl14_reg"
    path = "JeffreyXiang/TRELLIS-image-large"
    enc_pretrained = f"{path}/ckpts/slat_enc_swin8_B_64l8_fp16"

    NUM_EPOCHS = 10
    CODEBOOK_SIZE = 256
    CODEBOOK_DIM = 8
    BATCH_SIZE = 1024

    # init models
    encoder = models.from_pretrained(enc_pretrained).eval().cuda()
    vq = VectorQuantizerEMA(CODEBOOK_SIZE, CODEBOOK_DIM).train().cuda()
    
    # load data
    metadata = pd.read_csv(os.path.join(output_dir, 'metadata.csv'))
    metadata = metadata[metadata[f'feature_{feat_model}'] == True]

    # preprocess all features into a single tensor
    all_latents = []
    for sha256 in metadata['sha256'].values:
        feats = np.load(os.path.join(output_dir, 'features', feat_model, f'{sha256}.npz'))
        feats = sp.SparseTensor(
            feats = torch.from_numpy(feats['patchtokens']).float(),
            coords = torch.cat([
                torch.zeros(feats['patchtokens'].shape[0], 1).int(),
                torch.from_numpy(feats['indices']).int(),
            ], dim=1),
        ).cuda()
        with torch.no_grad():
            latent = encoder(feats, sample_posterior=False)
            all_latents.append(latent.feats.cpu())

    all_latents = torch.cat(all_latents, dim=0)
    
    if DEBUG:
        # print stats about encoder outputs
        print("Encoder output stats:")
        print(f"  Shape: {all_latents.shape}")
        print(f"  Mean: {all_latents.mean():.3f}")
        print(f"  Std: {all_latents.std():.3f}")
        print(f"  Min: {all_latents.min():.3f}")
        print(f"  Max: {all_latents.max():.3f}")
        print(f"  Dynamic range: {all_latents.max() - all_latents.min():.3f}")
    
    dataset = TensorDataset(all_latents)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # training loop
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        epoch_perplexity = 0
        epoch_usage = torch.zeros(CODEBOOK_SIZE, device='cuda')
        for batch in dataloader:
            latents = batch[0].cuda()
            loss, _, perplexity, encodings = vq(latents)
            epoch_loss += loss.item()
            epoch_perplexity += perplexity.item()
            epoch_usage += encodings.sum(0)
            
        avg_loss = epoch_loss / len(dataloader)
        avg_perplexity = epoch_perplexity / len(dataloader)
        dead_codes = (epoch_usage == 0).sum().item()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Commitment Loss: {avg_loss:.6f}")
        print(f"  Perplexity: {avg_perplexity:.2f}")
        print(f"  Dead codes: {dead_codes}/{CODEBOOK_SIZE}")

if __name__ == '__main__':
    main()
