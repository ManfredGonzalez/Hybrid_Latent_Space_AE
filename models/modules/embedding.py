import torch
import torch.nn as nn
import torch.nn.functional as F

class VQEmbedding(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=128, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings # Number of vectors in the codebook
        self.commitment_cost = commitment_cost # Beta, the commitment loss weight

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        #Initializes the embedding weights uniformly to help with training stability.
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings) 

    def forward(self, z):
        b, c, h, w = z.shape
        z_channel_last = z.permute(0, 2, 3, 1) # (B, H, W, C)
        z_flattened = z_channel_last.reshape(b*h*w, self.embedding_dim)

        # Calculate distances between z and the codebook embeddings |a-b|²
        # Efficient computation of Euclidean distances between the input vectors and codebook entries using the identity
        distances = (
            torch.sum(z_flattened ** 2, dim=-1, keepdim=True)                 # a²
            + torch.sum(self.embedding.weight.t() ** 2, dim=0, keepdim=True)  # b²
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())        # -2ab
        )

        # Get the index with the smallest distance
        # Vector Quantization: Selects the index of the closest codebook vector for each input patch (quantization step).
        encoding_indices = torch.argmin(distances, dim=-1)

        # Get the quantized vector
        # Codebook Lookup & Reshape
        # Codebook loss: Encourages codebook embeddings to match encoder outputs 
        # Commitment loss: Encourages encoder outputs to commit to codebook entries
        # Retrieves quantized vectors (z_q) from the codebook using the selected indices and reshapes them to the original z format.
        z_q = self.embedding(encoding_indices)
        z_q = z_q.reshape(b, h, w, self.embedding_dim)
        z_q = z_q.permute(0, 3, 1, 2)

        # Calculate the commitment loss
        mse_loss = nn.MSELoss(reduction="sum")

        commitment_loss = self.commitment_cost * mse_loss(z_q.detach(), z)
        codebook_loss = mse_loss(z_q, z.detach())

        loss = codebook_loss + commitment_loss

        # Straight-through estimator trick for gradient backpropagation
        # Ensures gradients flow from z_q to z during backpropagation while using quantized values for the forward pass.
        z_q = z + (z_q - z).detach()

        return z_q, loss, encoding_indices, commitment_loss, codebook_loss