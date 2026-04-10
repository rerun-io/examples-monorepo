
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter

SUM_FREQ: int = 100

class Logger:
    def __init__(self, name: str, scheduler: LRScheduler) -> None:
        self.total_steps: int = 0
        self.running_loss: dict[str, float] = {}
        self.writer: SummaryWriter | None = None
        self.name: str = name
        self.scheduler: LRScheduler = scheduler

    def _print_training_status(self) -> None:
        if self.writer is None:
            self.writer = SummaryWriter(f"runs/{self.name}")
            print([k for k in self.running_loss])

        lr: float = self.scheduler.get_lr().pop()
        metrics_data: list[float] = [self.running_loss[k]/SUM_FREQ for k in self.running_loss.keys()]
        training_str: str = f"[{self.total_steps+1:6d}, {lr:10.7f}] "
        metrics_str: str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)

        # print the training status
        print(training_str + metrics_str)

        for key in self.running_loss:
            val: float = self.running_loss[key] / SUM_FREQ
            self.writer.add_scalar(key, val, self.total_steps)
            self.running_loss[key] = 0.0

    def push(self, metrics: dict[str, float]) -> None:

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

        self.total_steps += 1

    def write_dict(self, results: dict[str, float]) -> None:
        if self.writer is None:
            self.writer = SummaryWriter(f"runs/{self.name}")
            print([k for k in self.running_loss])

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self) -> None:
        self.writer.close()
