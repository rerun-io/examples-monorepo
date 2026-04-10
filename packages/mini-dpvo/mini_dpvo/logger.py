"""TensorBoard training logger for DPVO.

Provides a lightweight :class:`Logger` that accumulates per-step metrics,
computes running averages, and periodically writes summaries to TensorBoard.
This is used during *training* of the VONet model (not during inference).
"""

from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter

# Number of training steps between each log print / TensorBoard write.
SUM_FREQ: int = 100


class Logger:
    """Accumulates training metrics and writes periodic TensorBoard summaries.

    Metrics are accumulated over ``SUM_FREQ`` steps, then averaged and
    written to TensorBoard.  The current learning rate is printed alongside
    the averaged metrics for quick monitoring.

    Attributes:
        total_steps: Running count of training iterations seen so far.
        running_loss: Dictionary mapping metric names to their accumulated
            (un-averaged) values since the last print.
        writer: Lazily initialised TensorBoard ``SummaryWriter``.
        name: Experiment name used as the TensorBoard run directory
            (``runs/{name}``).
        scheduler: Learning-rate scheduler, queried for the current LR
            when printing status.
    """

    def __init__(self, name: str, scheduler: LRScheduler) -> None:
        self.total_steps: int = 0
        self.running_loss: dict[str, float] = {}
        self.writer: SummaryWriter | None = None
        self.name: str = name
        self.scheduler: LRScheduler = scheduler

    def _print_training_status(self) -> None:
        """Print averaged metrics and write them to TensorBoard.

        Called internally every ``SUM_FREQ`` steps.  On first invocation the
        TensorBoard writer is created lazily.
        """
        if self.writer is None:
            self.writer = SummaryWriter(f"runs/{self.name}")
            print([k for k in self.running_loss])

        lr: float = self.scheduler.get_lr().pop()
        metrics_data: list[float] = [self.running_loss[k]/SUM_FREQ for k in self.running_loss]
        training_str: str = f"[{self.total_steps+1:6d}, {lr:10.7f}] "
        metrics_str: str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)

        # print the training status
        print(training_str + metrics_str)

        for key in self.running_loss:
            val: float = self.running_loss[key] / SUM_FREQ
            self.writer.add_scalar(key, val, self.total_steps)
            self.running_loss[key] = 0.0

    def push(self, metrics: dict[str, float]) -> None:
        """Record one training step's metrics.

        Accumulates the given metric values.  Every ``SUM_FREQ`` steps the
        running averages are printed and written to TensorBoard, and the
        accumulators are reset.

        Args:
            metrics: Dictionary mapping metric names (e.g. ``"loss"``,
                ``"flow_loss"``) to their scalar values for this step.
        """
        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

        self.total_steps += 1

    def write_dict(self, results: dict[str, float]) -> None:
        """Write a dictionary of scalar results to TensorBoard immediately.

        Unlike :meth:`push`, this does not accumulate -- each value is
        written as a scalar summary at the current ``total_steps``.

        Args:
            results: Dictionary mapping metric names to scalar values.
        """
        if self.writer is None:
            self.writer = SummaryWriter(f"runs/{self.name}")
            print([k for k in self.running_loss])

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self) -> None:
        """Flush and close the TensorBoard writer."""
        self.writer.close()
