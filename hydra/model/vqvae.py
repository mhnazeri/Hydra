import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from accelerate import PartialState
import numpy as np
from sklearn.cluster import KMeans


def calculate_vq_metrics(indices, num_embeddings):
    """
    indices: [B*T, 1] - the indices selected from the codebook
    num_embeddings: K (size of codebook)
    """
    # Calculate frequency of each code
    # Flatten and count occurrences
    flat_indices = indices.view(-1)
    counts = torch.bincount(flat_indices, minlength=num_embeddings).float()

    # Calculate probabilities
    probs = counts / torch.sum(counts)

    perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))

    usage = (counts > 0).sum().float() / num_embeddings * 100

    return perplexity.item(), usage.item()


class EMAVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5, reset_freq=100, name=None, temperature=0.0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.reset_freq = reset_freq
        self.name = name if name is not None else self.__class__.__name__
        self.temperature = temperature

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.normal_()

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', torch.empty(num_embeddings, embedding_dim))
        self.register_buffer('_step', torch.tensor(0, dtype=torch.long))
        self.register_buffer('_is_initialized', torch.tensor(0, dtype=torch.uint8))
        self._ema_w.data.normal_()

    def forward(self, inputs):
        # inputs: [B*T, D] — raw encoder output, arbitrary scale
        state = PartialState()

        flat_input = inputs.view(-1, self.embedding_dim)
        flat_input = F.normalize(flat_input, p=2, dim=1)   # unit hypersphere

        embedding_weight = F.normalize(self.embedding.weight, p=2, dim=1)

        # Distances and nearest codes
        similarity = torch.matmul(flat_input, embedding_weight.t())

        if self.training and self.temperature > 0.0:
            logits = similarity / self.temperature
            probs = F.softmax(logits, dim=-1)
            dist_sampler = torch.distributions.Categorical(probs)
            encoding_indices = dist_sampler.sample()
        else:
            encoding_indices = torch.argmax(similarity, dim=1)

        encodings = F.one_hot(encoding_indices, self.num_embeddings).type_as(flat_input)

        # Quantize (result is a normalized vector, same space as flat_input)
        quantized = torch.matmul(encodings, embedding_weight)  # [N, D], normalized

        # EMA update
        if self.training:
            encodings_sum = encodings.sum(0)
            dw = torch.matmul(encodings.t(), flat_input)

            if state.num_processes > 1:
                dist.all_reduce(encodings_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(dw, op=dist.ReduceOp.SUM)

            self._ema_cluster_size.data.mul_(self.decay).add_(
                encodings_sum, alpha=1 - self.decay)

            # Laplace smoothing
            n = self._ema_cluster_size.data.sum()
            self._ema_cluster_size.data = (
                (self._ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n)

            self._ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            # Re-normalize after averaging
            self.embedding.weight.data.copy_(
                F.normalize(self._ema_w / self._ema_cluster_size.unsqueeze(1), p=2, dim=1))

            self._step += 1
            if self._step % self.reset_freq == 0:
                self.reset_dead_codes(flat_input)   # pass already-normalized input

        # Commitment loss
        e_latent_loss = F.mse_loss(quantized.detach(), flat_input)
        loss = self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized_st = flat_input + (quantized - flat_input).detach()
        quantized_st = quantized_st.view(inputs.shape)

        # Perplexity
        with torch.no_grad():
            probs = encodings.sum(0)
            if state.num_processes > 1:
                dist.all_reduce(probs, op=dist.ReduceOp.SUM)
            probs = probs / (probs.sum() + 1e-10)
            perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))
            usage = (self._ema_cluster_size > 0.5).float().mean() * 100

        return quantized_st, loss, {
            "perplexity": perplexity.item(),
            "usage": usage.item(),
        }

    @torch.no_grad()
    def reset_dead_codes(self, flat_input):
        """
        flat_input: [N, D]
        """
        state = PartialState()
        dead_indices = (self._ema_cluster_size < 1.0).nonzero(as_tuple=True)[0]

        if len(dead_indices) == 0:
            return

        if state.process_index == 0:
            print(f"\n[{self.name}] Resetting {len(dead_indices)} dead codes at step {self._step.item()}")

        if state.num_processes > 1:
            gathered = [torch.zeros_like(flat_input) for _ in range(state.num_processes)]
            dist.all_gather(gathered, flat_input)
            flat_input = torch.cat(gathered, dim=0)
            perm = torch.randperm(flat_input.size(0), device=flat_input.device)
            dist.broadcast(perm, src=0)
        else:
            perm = torch.randperm(flat_input.size(0), device=flat_input.device)

        actual_resets = min(len(dead_indices), flat_input.size(0))
        target_indices = dead_indices[:actual_resets]
        new_embeddings = flat_input[perm[:actual_resets]]

        self.embedding.weight.data[target_indices] = new_embeddings
        self._ema_w.data[target_indices] = new_embeddings
        self._ema_cluster_size.data[target_indices] = 1.0
