
import torch
from torch import Tensor


def fps(src: Tensor, batch: Tensor, ratio: float = 0.5, random_start: bool = True) -> Tensor:
    """Pure PyTorch Farthest Point Sampling — drop-in replacement for torch_cluster.fps."""
    device = src.device
    N = src.shape[0]

    if batch is None:
        batch = torch.zeros(N, dtype=torch.long, device=device)

    indices_out = []
    for b in batch.unique():
        mask = batch == b
        pts = src[mask]
        n_pts = pts.shape[0]
        n_sample = max(1, int(n_pts * ratio))

        global_indices = mask.nonzero(as_tuple=True)[0]

        if random_start:
            first = torch.randint(0, n_pts, (1,), device=device).item()
        else:
            first = 0

        sampled_local = [first]
        dist = torch.norm(pts - pts[first].unsqueeze(0), dim=-1)

        for _ in range(n_sample - 1):
            farthest = dist.argmax().item()
            sampled_local.append(farthest)
            new_dist = torch.norm(pts - pts[farthest].unsqueeze(0), dim=-1)
            dist = torch.minimum(dist, new_dist)

        sampled_local = torch.tensor(sampled_local, device=device, dtype=torch.long)
        indices_out.append(global_indices[sampled_local])

    return torch.cat(indices_out, dim=0)
