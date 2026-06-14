import random
from collections import deque

import torch
from torch import Tensor

from .base import BaseOrdering


class BFSOrdering(BaseOrdering):
    """Order faces by BFS traversal starting from a random face.

    Disconnected components are handled by restarting BFS from the next
    unvisited face after each component is exhausted.
    """

    def permute(self, faces: Tensor, face_neighbors: Tensor, lengths: Tensor) -> Tensor:
        B, N, _ = faces.shape
        device = faces.device
        perm = self._base_perm(B, N, lengths, device)

        for b in range(B):
            L = int(lengths[b].item())
            adj = face_neighbors[b, :L].tolist()   # list of 3-lists (tri) or 4-lists (quad)

            visited = [False] * L
            order = []
            start = random.randrange(L)

            # BFS from random start; restart for disconnected components
            seeds = [start] + list(range(L))
            for seed in seeds:
                if visited[seed]:
                    continue
                visited[seed] = True
                queue = deque([seed])
                while queue:
                    node = queue.popleft()
                    order.append(node)
                    for nb in adj[node]:
                        if nb >= 0 and not visited[nb]:
                            visited[nb] = True
                            queue.append(nb)

            perm[b, :L] = torch.tensor(order, dtype=torch.long, device=device)

        return perm
