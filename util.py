import math


class LRCosineScheduler:
    def __init__(
        self,
        optimizer,
        warmup_iters: int,
        decay_iters: int,
        start_lr: float,
        min_lr: float,
        max_lr: float,
        verbose: bool = False,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.decay_iters = decay_iters
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.start_lr = start_lr
        self.verbose = verbose
        self.iter = 1

    def step(self) -> None:
        if self.iter < self.warmup_iters:
            lr = (
                self.max_lr - self.start_lr
            ) / self.warmup_iters * self.iter + self.start_lr
        elif self.iter > self.decay_iters:
            lr = self.min_lr
        else:
            decay = (self.iter - self.warmup_iters) / (
                self.decay_iters - self.warmup_iters
            )
            decay = min(decay, 1.0)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
            lr = self.min_lr + coeff * (self.max_lr - self.min_lr)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        if self.verbose:
            print(self.iter, lr)
        self.iter += 1


def cos_anneal(e0, e1, t0, t1, e) -> float:
    alpha = max(
        0, min(1, (e - e0) / (e1 - e0))
    )  # what fraction of the way through are we
    alpha = 1.0 - math.cos(alpha * math.pi / 2)  # warp through cosine
    t = alpha * t1 + (1 - alpha) * t0  # interpolate accordingly
    return t
