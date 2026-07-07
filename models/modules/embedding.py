import torch
import torch.nn as nn
import torch.nn.functional as F

class VQEmbedding(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=128, commitment_cost=0.25, reduction='sum', l2_normalize=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings # Number of vectors in the codebook
        self.commitment_cost = commitment_cost # Beta, the commitment loss weight
        self.reduction = reduction # How to reduce the loss: 'sum' or 'mean'
        self.l2_normalize = l2_normalize # Normalize codes/lookup to the unit sphere before the distance computation (helps codebook utilization at low dims)

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        #Initializes the embedding weights uniformly to help with training stability.
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

        # Codebook health stats from the most recent forward pass (diagnostics only,
        # not used in any loss). perplexity == exp(entropy of code usage): num_embeddings
        # when every code is used equally often, 1 when only one code is ever picked.
        # codebook_usage == fraction of codes used at least once in the batch.
        self.register_buffer('perplexity', torch.zeros(()), persistent=False)
        self.register_buffer('codebook_usage', torch.zeros(()), persistent=False)

    @torch.no_grad()
    def init_from_data(self, z, num_iters=10):
        """Re-seed the codebook via k-means (Lloyd's algorithm) on encoder activations.

        z: encoder output with the same (B, C, H, W) layout `forward` expects, e.g. from
        one batch pushed through the encoder (in fp32, outside autocast) before training
        starts. Replaces the default uniform init with centroids of the actual data
        distribution, which fixes the scale mismatch against the codebook's uniform
        [-1/N, 1/N] init and cuts down on dead codes early in training.
        """
        b, c, h, w = z.shape
        z_flattened = z.permute(0, 2, 3, 1).reshape(b * h * w, self.embedding_dim)
        n_vectors = z_flattened.shape[0]

        if n_vectors < self.num_embeddings:
            raise ValueError(
                f"init_from_data needs at least num_embeddings ({self.num_embeddings}) vectors "
                f"to seed centroids, got {n_vectors}. Pass a larger batch."
            )

        # Seed centroids from the data itself (k-means++ style would be nicer, but plain
        # random seeding is enough at these codebook sizes and converges in a few iters).
        perm = torch.randperm(n_vectors, device=z_flattened.device)[:self.num_embeddings]
        centroids = z_flattened[perm].clone()

        for _ in range(num_iters):
            # Assign in the same metric `forward` uses for lookup (cosine when
            # l2_normalize, else Euclidean), but always update/store raw centroids --
            # mirrors forward(), which compares normalized vectors but returns the raw
            # codeword.
            if self.l2_normalize:
                z_cmp = F.normalize(z_flattened, dim=-1)
                cb_cmp = F.normalize(centroids, dim=-1)
                distances = 2.0 - 2.0 * torch.matmul(z_cmp, cb_cmp.t())
            else:
                distances = (
                    torch.sum(z_flattened ** 2, dim=-1, keepdim=True)
                    + torch.sum(centroids ** 2, dim=-1).unsqueeze(0)
                    - 2 * torch.matmul(z_flattened, centroids.t())
                )
            assignments = torch.argmin(distances, dim=-1)

            for k in range(self.num_embeddings):
                mask = assignments == k
                if mask.any():
                    centroids[k] = z_flattened[mask].mean(dim=0)
                else:
                    # Empty cluster: teleport to a random data point instead of freezing
                    # it in a dead region, so it gets a chance to win points next pass.
                    centroids[k] = z_flattened[torch.randint(n_vectors, (1,), device=z_flattened.device)].squeeze(0)

        self.embedding.weight.data.copy_(centroids)

    def forward(self, z):
        b, c, h, w = z.shape
        z_channel_last = z.permute(0, 2, 3, 1) # (B, H, W, C)
        z_flattened = z_channel_last.reshape(b*h*w, self.embedding_dim)

        codebook = self.embedding.weight

        if self.l2_normalize:
            # Cosine-similarity nearest-neighbor search only: for unit vectors
            # ||a-b||^2 = 2 - 2*a.b. The returned z_q below is still the RAW
            # codeword and z is untouched, so the rest of the pipeline (residual
            # addition, SWD, decoder) sees the same raw-scale space as when
            # l2_normalize is off -- only which code gets picked changes.
            z_cmp = F.normalize(z_flattened, dim=-1)
            cb_cmp = F.normalize(codebook, dim=-1)
            distances = 2.0 - 2.0 * torch.matmul(z_cmp, cb_cmp.t())
        else:
            # Calculate distances between z and the codebook embeddings |a-b|²
            # Efficient computation of Euclidean distances between the input vectors and codebook entries using the identity
            distances = (
                torch.sum(z_flattened ** 2, dim=-1, keepdim=True)                 # a²
                + torch.sum(codebook.t() ** 2, dim=0, keepdim=True)  # b²
                - 2 * torch.matmul(z_flattened, codebook.t())        # -2ab
            )

        # Get the index with the smallest distance
        # Vector Quantization: Selects the index of the closest codebook vector for each input patch (quantization step).
        encoding_indices = torch.argmin(distances, dim=-1)

        # Codebook health diagnostics, detached: how many distinct codes fired this
        # batch, and how uniformly. Cheap to compute (num_embeddings-sized histogram).
        with torch.no_grad():
            one_hot = F.one_hot(encoding_indices, self.num_embeddings).float()
            avg_probs = one_hot.mean(dim=0)
            self.perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            self.codebook_usage = (avg_probs > 0).float().mean()

        # Get the quantized vector
        # Codebook Lookup & Reshape
        # Codebook loss: Encourages codebook embeddings to match encoder outputs
        # Commitment loss: Encourages encoder outputs to commit to codebook entries
        # Retrieves quantized vectors (z_q) from the codebook using the selected indices and reshapes them to the original z format.
        z_q = codebook[encoding_indices]
        z_q = z_q.reshape(b, h, w, self.embedding_dim)
        z_q = z_q.permute(0, 3, 1, 2)

        # Calculate the commitment loss
        mse_loss = nn.MSELoss(reduction=self.reduction)

        commitment_loss = self.commitment_cost * mse_loss(z_q.detach(), z)
        codebook_loss = mse_loss(z_q, z.detach())

        loss = codebook_loss + commitment_loss

        # Straight-through estimator trick for gradient backpropagation
        # Ensures gradients flow from z_q to z during backpropagation while using quantized values for the forward pass.
        z_q = z + (z_q - z).detach()

        return z_q, loss, encoding_indices, commitment_loss, codebook_loss