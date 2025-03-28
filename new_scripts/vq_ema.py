### Implementation of Vector Quantizer with EMA
### Modified from: https://github.com/zalandoresearch/pytorch-vq-vae

import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizerEMA(nn.Module):
    # commitment_cost would depend on the scale of reconstruction loss
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.normal_()
        
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())

    def forward(self, inputs):
        N, D = inputs.shape
        assert D == self.embedding_dim
                
        distances = torch.cdist(inputs, self.embedding.weight, p=2)
            
        encoding_indices = torch.argmin(distances, dim=1, keepdim=True)
        encodings = torch.zeros(N, self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embedding.weight)
        
        # Use EMA to update the embedding vectors
        if self.training:
            # Update cluster size
            ema_cluster_size = self.ema_cluster_size * self.decay + \
                                     (1 - self.decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(ema_cluster_size).item()
            ema_cluster_size = (
                (ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n)
            
            # Update embeddings sum
            dw = torch.matmul(encodings.t(), inputs)
            ema_w = self.ema_w * self.decay + (1 - self.decay) * dw
            
            self.ema_cluster_size.copy_(ema_cluster_size)
            self.ema_w.copy_(ema_w)
            
            normalized_ema_w = ema_w / ema_cluster_size.unsqueeze(1)
            self.embedding.weight.data.copy_(normalized_ema_w)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Calculate perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return loss, quantized, perplexity, encodings
    
if __name__ == "__main__":
    vq_ema = VectorQuantizerEMA(12, 8)
    tens = torch.randn(8028, 8)
    loss, quantized, perplexity, encodings = vq_ema(tens)
    print(loss, quantized.shape, perplexity, encodings.shape)
