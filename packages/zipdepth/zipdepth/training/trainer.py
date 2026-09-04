import ctypes
import gc
import glob
import os
import time
from contextlib import nullcontext, suppress
from typing import Literal, TypeAlias

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
from tqdm import tqdm

try:
    import wandb  # noqa: F401
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from monopriors.models.zipdepth_checkpoint import RANGE_MARGIN_M_KEY
from monopriors.third_party.zipdepth.model_utils import strip_state_dict_prefixes

from zipdepth.loss import MetricDepthLoss, ZipDepthLoss
from zipdepth.training.visualization import depth_to_spectral

CompileMode: TypeAlias = Literal['off', 'default', 'reduce-overhead', 'max-autotune']


def trim_memory():
    with suppress(Exception):
        ctypes.CDLL("libc.so.6").malloc_trim(0)


# =============================================================================
# ZIPDEPTH TRAINER
# =============================================================================

class ZipDepthTrainer:
    """
    ZipDepth trainer for monocular depth estimation.
    Supports single-GPU and distributed training with mixed precision.
    """

    def __init__(self,
                student,
                train_loader,
                optimizer,
                scheduler,
                device,
                use_amp=True,
                log_wandb=False,
                writer=None,
                is_distributed=False,
                rank=0,
                world_size=1,
                use_profiler=False,
                profile_dir='./log/profiler',
                profile_wait=2,
                profile_warmup=2,
                profile_active=5,
                profile_repeat=1,
                amp_dtype='bfloat16',
                alpha_ssi: float = 1.0,
                alpha_grad: float = 2.0,
                target_mode: Literal['ssi', 'metric'] = 'ssi',
                metric_gradient_weight: float = 0.5,
                pin_batchnorm_eval: bool = False,
                compile_mode: CompileMode = 'reduce-overhead',
                channels_last: bool = False,
                range_margin_m: float = 0.0,
                ):
        """
        Args:
            student: Model to train
            train_loader: Training data loader
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler (optional)
            device: Training device
            use_amp: Enable automatic mixed precision training
            log_wandb: Enable Weights & Biases logging
            writer: TensorBoard SummaryWriter instance (optional)
            is_distributed: Whether running in distributed mode
            rank: Process rank for distributed training
            world_size: Total number of processes
            use_profiler: Enable torch.profiler for op-level + H2D transfer analysis (default False)
            profile_dir: Output directory for profiler traces (TensorBoard + chrome JSON)
            profile_wait: Profiler schedule — steps to skip before warmup
            profile_warmup: Profiler schedule — steps to warm up (not recorded)
            profile_active: Profiler schedule — steps to actually record
            profile_repeat: How many times to repeat the profiling cycle (0 = unlimited)
            amp_dtype: AMP dtype — 'bfloat16' (recommended on Ampere, no GradScaler needed)
                       or 'float16' (requires GradScaler, narrower range)
            alpha_ssi: Weight for SSI loss component
            alpha_grad: Weight for gradient loss component
            target_mode: Relative SSI or prompted metric training lane
            metric_gradient_weight: Weight on the multi-scale gradient term in the metric
                lane. Upstream ZipDepth's SSI lane uses 2.0; 0.5 leaves the term at ~9%
                of the objective, which under-penalizes depth-discontinuity errors.
            pin_batchnorm_eval: Return BatchNorm modules to eval after entering train mode
            compile_mode: torch.compile mode, or 'off'; dev-mode beartype disables it
            channels_last: Move four-dimensional batches with channels-last strides
            range_margin_m: Prompted head output-range margin of the student, in metres,
                recorded in every checkpoint so inference rebuilds the same head
        """
        self.student = student.to(device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp
        self.log_wandb = log_wandb and WANDB_AVAILABLE
        self.writer = writer
        self.global_step = 0
        self.training_stage = 0
        self.is_distributed = is_distributed
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)
        self.use_profiler = use_profiler
        self.profile_dir = profile_dir
        self.profile_wait = profile_wait
        self.profile_warmup = profile_warmup
        self.profile_active = profile_active
        self.profile_repeat = profile_repeat
        self.amp_dtype = torch.bfloat16 if amp_dtype == 'bfloat16' else torch.float16
        self._tracemalloc_started = False
        self._snapshot_start = None
        self._restart_interval = 1000
        self._loader_config = None
        if target_mode not in ('ssi', 'metric'):
            raise ValueError(f"unknown target mode {target_mode!r}")
        self.target_mode = target_mode
        self.pin_batchnorm_eval = pin_batchnorm_eval
        self.channels_last = channels_last
        self.range_margin_m = range_margin_m

        # Loss function initialization
        self.criterion: nn.Module = (
            MetricDepthLoss(gradient_weight=metric_gradient_weight)
            if target_mode == 'metric'
            else ZipDepthLoss(alpha_ssi=alpha_ssi, alpha_grad=alpha_grad)
        ).to(device)

        # GradScaler only needed for FP16 — BF16 has FP32-range exponents so no overflow
        self.scaler = (
            torch.amp.GradScaler('cuda')
            if use_amp and self.amp_dtype == torch.float16
            else None
        )

        # Compile student for faster iteration (PyTorch >= 2.0)
        # Under the beartype dev env, Dynamo tracing through beartype's jaxtyping wrapper
        # raises a desynchronization error; production environments still compile.
        if compile_mode != 'off' and os.environ.get("PIXI_DEV_MODE") != "1":
            self.student = torch.compile(self.student, mode=compile_mode)

    def train(self, num_epochs: int, save_dir: str = './checkpoints', start_epoch: int = 0,
            save_every_steps: int = 0, max_step_checkpoints: int = 5, max_steps: int = 0,
            skip_batches: int | None = None):
        if self.is_main:
            os.makedirs(save_dir, exist_ok=True)

        self.save_dir = save_dir
        self.save_every_steps = save_every_steps
        self.max_step_checkpoints = max_step_checkpoints

        steps_per_epoch = len(self.train_loader)

        # When resuming mid-epoch, skip the batches already processed so the
        # scheduler is stepped exactly the right number of times.
        initial_skip = skip_batches if skip_batches is not None else self.global_step % steps_per_epoch
        if initial_skip > 0 and self.is_main:
            print(f"\nResume: skipping first {initial_skip}/{steps_per_epoch} batches "
                  f"of epoch {start_epoch} (global_step={self.global_step})")

        # Total iterations for the global progress bar
        total_iterations = steps_per_epoch * (num_epochs - start_epoch) - initial_skip
        if max_steps > 0:
            total_iterations = min(total_iterations, max_steps - self.global_step)

        disable_tqdm = self.is_distributed and not self.is_main
        pbar = tqdm(
            total=total_iterations,
            desc='Training',
            disable=disable_tqdm,
            initial=0,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )

        train_loss = 0.0
        for epoch in range(start_epoch, num_epochs):
            skip = initial_skip if epoch == start_epoch else 0
            train_loss = self.train_epoch(epoch, num_epochs, pbar,
                                          skip_batches=skip, max_steps=max_steps)

            if self.is_main:
                print(f'\nEpoch {epoch}: Train Loss = {train_loss:.4f}')
                self.save_checkpoint(os.path.join(save_dir, f'epoch_{epoch}.pth'), epoch, train_loss)

            if max_steps > 0 and self.global_step >= max_steps:
                if self.is_main:
                    print(f'Reached max_steps={max_steps:,}. Stopping.')
                break

        pbar.close()

        if self.is_main:
            self.save_checkpoint(os.path.join(save_dir, 'final_model.pth'), num_epochs - 1, train_loss)


    def _build_profiler(self):
        """Build a torch.profiler instance if profiling is enabled, else return nullcontext."""
        if not self.use_profiler or not self.is_main:
            return nullcontext()

        os.makedirs(self.profile_dir, exist_ok=True)

        def on_trace_ready(p):
            tensorboard_trace_handler(self.profile_dir)(p)
            trace_files = sorted(
                [f for f in os.listdir(self.profile_dir) if f.endswith('.json') or f.endswith('.json.gz')],
                key=lambda f: os.path.getmtime(os.path.join(self.profile_dir, f)),
            )
            latest = os.path.join(self.profile_dir, trace_files[-1]) if trace_files else self.profile_dir

            # --- per-op table (top 30 by CUDA time) ---
            avgs = p.key_averages()
            try:
                table_str = avgs.table(sort_by='cuda_time_total', row_limit=30)
            except Exception:
                table_str = avgs.table(sort_by='cpu_time_total', row_limit=30)

            _cuda_attr = None
            for _candidate in ('self_cuda_time_total', 'cuda_time_total', 'device_time_total'):
                if any(getattr(e, _candidate, 0) > 0 for e in avgs):
                    _cuda_attr = _candidate
                    break

            def _cuda_us(e):
                return getattr(e, _cuda_attr, 0) if _cuda_attr else 0

            # --- simple bottleneck summary ---
            total_cuda_us = sum(_cuda_us(e) for e in avgs)
            total_cpu_us  = sum(getattr(e, 'cpu_time_total', 0) for e in avgs)
            h2d_us = sum(_cuda_us(e) for e in avgs
                         if 'to_copy' in e.key or 'copy_' in e.key)
            top3 = sorted(avgs, key=_cuda_us, reverse=True)[:3]

            lines = [
                '',
                '=' * 60,
                'PROFILER SUMMARY',
                '=' * 60,
                f'  Total CUDA time : {total_cuda_us/1e3:.1f} ms',
                f'  Total CPU time  : {total_cpu_us/1e3:.1f} ms',
                f'  H2D transfers   : {h2d_us/1e3:.1f} ms  ({100*h2d_us/max(total_cuda_us,1):.1f}% of CUDA)',
                '',
                '  Top-3 ops by CUDA time:',
            ]
            for i, e in enumerate(top3, 1):
                lines.append(f'    {i}. {e.key:<40s}  {_cuda_us(e)/1e3:7.2f} ms')

            # bottleneck verdict
            if total_cpu_us > total_cuda_us * 2:
                verdict = 'CPU-BOUND  -> dataloader / preprocessing is the bottleneck'
            elif h2d_us > total_cuda_us * 0.15:
                verdict = 'H2D-BOUND  -> host->device transfers are eating >15% of CUDA time'
            else:
                verdict = 'GPU-BOUND  -> compute is the bottleneck (good)'
            lines += ['', f'  Verdict: {verdict}', '=' * 60, '']

            summary = '\n'.join(lines)
            print(summary)
            print(table_str)

            ts = time.strftime('%Y%m%d_%H%M%S')
            txt_path = os.path.join(self.profile_dir, f'profiler_summary_{ts}.txt')
            with open(txt_path, 'w') as f:
                f.write(summary + '\n\n' + table_str)

            print(f'[Profiler] Trace   -> {latest}')
            print(f'           Summary -> {txt_path}')
            print(f'           TensorBoard : tensorboard --logdir {self.profile_dir}')
            print('           Perfetto    : https://ui.perfetto.dev  (load the JSON)')

        return profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=self.profile_wait,
                warmup=self.profile_warmup,
                active=self.profile_active,
                repeat=self.profile_repeat,
            ),
            on_trace_ready=on_trace_ready,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

    def train_epoch(self, epoch: int, num_epochs: int, pbar: tqdm,
                    skip_batches: int = 0, max_steps: int = 0) -> float:
        """Train for one epoch.

        Args:
            skip_batches: Number of leading batches to skip (for mid-epoch resume).
                          The sampler must reproduce the same order as the original
                          run, which is guaranteed when set_epoch() is called with
                          the same epoch value.
        """

        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)

        self.student.train()
        if self.pin_batchnorm_eval:
            for module in self.student.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        processed_batches = 0

        disable_tqdm = self.is_distributed and not self.is_main

        with self._build_profiler() as prof:
            for batch_idx, batch in enumerate(self.train_loader):

                if batch_idx < skip_batches:
                    pbar.update(1)
                    continue

                memory_format: torch.memory_format = torch.channels_last if self.channels_last else torch.preserve_format
                images = batch['image'].to(self.device, non_blocking=True, memory_format=memory_format)
                images = images.float() / 255.0

                prompt_depth = None
                if self.target_mode == 'metric':
                    teacher_depth = batch['target_depth'].to(self.device, non_blocking=True, memory_format=memory_format).float()
                    mask = batch['target_valid'].to(self.device, non_blocking=True, memory_format=memory_format)
                    # Feed the RAW prompt, exactly what the teacher saw when it produced
                    # these labels (promptda_stream.py:254). Zeroing low-confidence pixels
                    # made the student solve a different problem from the teacher, and the
                    # degenerate cases it guarded against are already handled inside the
                    # model, which computes its own finite/0.1-4 m validity for the
                    # normalization range and clamps the span.
                    prompt_depth = batch['prompt_depth'].to(self.device, non_blocking=True, memory_format=memory_format).float()
                else:
                    teacher_depth = batch['depth'].to(self.device, non_blocking=True, memory_format=memory_format)
                    teacher_depth = teacher_depth.float().div_(256.0)

                    # Local addition: sparse catalog targets carry an explicit validity mask.
                    mask = batch.get('mask')
                    if mask is not None:
                        mask = mask.to(self.device, non_blocking=True, memory_format=memory_format).float()

                del batch

                with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                    student_depth = self.student(images, prompt_depth) if prompt_depth is not None else self.student(images)
                    loss, loss_dict = self.criterion(
                        pred=student_depth,
                        target=teacher_depth,
                        mask=mask,
                    )

                if self.scaler is not None:
                    # FP16: use GradScaler to prevent overflow
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0, foreach=True)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # BF16 / FP32: no overflow risk, direct backward
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0, foreach=True)
                    self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.optimizer.zero_grad(set_to_none=True)

                # Sync the host only after the whole step is enqueued: an .item() before
                # backward() drains the CUDA queue on the critical path and costs the
                # overlap that hides the next batch's data wait.
                loss_value = loss.item()
                loss_dict = {key: float(value) for key, value in loss_dict.items()}

                self.global_step += 1
                total_loss += loss_value
                processed_batches += 1

                if max_steps > 0 and self.global_step >= max_steps:
                    if not disable_tqdm:
                        pbar.update(1)
                    break

                avg_loss = total_loss / processed_batches
                primary_name = 'l1' if self.target_mode == 'metric' else 'ssi'

                if not disable_tqdm:
                    pbar.update(1)
                    pbar.set_postfix({
                        'E': f"{epoch+1}/{num_epochs}",
                        'B': f"{batch_idx+1}/{num_batches}",
                        'loss': f"{loss_value:.3f}",
                        'avg_loss': f"{avg_loss:.3f}",
                        primary_name: f"{loss_dict[primary_name]:.3f}",
                    })

                # Logging
                if self.is_main and self.writer is not None:
                    if self.global_step % 750 == 0:
                        self.writer.add_scalar('train/loss', loss_value, self.global_step)
                        self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                        self.writer.add_scalar(f'train/loss_{primary_name}', loss_dict[primary_name], self.global_step)
                        self.writer.add_scalar('train/loss_grad', loss_dict['grad'], self.global_step)

                    if self.global_step % 750 == 0:
                        n_show = min(6, images.size(0))

                        with torch.no_grad():
                            imgs_cpu = images[:n_show].detach().clone().cpu()
                            gt_cpu = teacher_depth[:n_show].detach().clone().cpu()
                            pred_cpu = student_depth[:n_show].detach().clone().cpu()

                            gt_depth_imgs = depth_to_spectral(gt_cpu)
                            depth_imgs = depth_to_spectral(pred_cpu)

                        self.writer.add_images('train/img', imgs_cpu, self.global_step)
                        self.writer.add_images('train/gt', gt_depth_imgs, self.global_step)
                        self.writer.add_images('train/pred', depth_imgs, self.global_step)
                        self.writer.flush()

                        del imgs_cpu, gt_cpu, pred_cpu, gt_depth_imgs, depth_imgs

                if self.save_every_steps > 0 and self.global_step % self.save_every_steps == 0 and self.is_main:
                    ckpt_path = os.path.join(self.save_dir, f'step_{self.global_step}.pth')
                    self.save_checkpoint(ckpt_path, epoch, loss_value)
                    print(f'\n  -> Checkpoint saved at step {self.global_step}')
                    self.cleanup_step_checkpoints()

                del student_depth, teacher_depth, mask, prompt_depth, loss

                # Advance profiler schedule (no-op after schedule completes or when profiling disabled)
                if prof is not None:
                    prof.step()

                if batch_idx % 500 == 0:
                    gc.collect()
                    trim_memory()

        return total_loss / max(processed_batches, 1)

    def save_checkpoint(self, path, epoch, loss):
        if not self.is_main:
            return

        raw_state = self.student.module.state_dict() if self.is_distributed else self.student.state_dict()
        model_state = strip_state_dict_prefixes(raw_state)

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'training_stage': self.training_stage,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            RANGE_MARGIN_M_KEY: self.range_margin_m,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)


    def cleanup_step_checkpoints(self):
        if not self.is_main:
            return

        step_ckpts = sorted(
            glob.glob(os.path.join(self.save_dir, 'step_*.pth')),
            key=lambda x: int(x.split('_')[-1].replace('.pth', ''))
        )

        if len(step_ckpts) > self.max_step_checkpoints:
            for old_ckpt in step_ckpts[:-self.max_step_checkpoints]:
                os.remove(old_ckpt)

    def load_checkpoint(self, path: str) -> tuple:
        """Load a training checkpoint. Returns (epoch, loss)."""
        from zipdepth.training.distributed import fix_state_dict_prefix
        ckpt = torch.load(path, map_location=self.device)
        sd   = fix_state_dict_prefix(ckpt['model_state_dict'], self.is_distributed)
        self.student.load_state_dict(sd)
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if self.scheduler and ckpt.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.global_step = ckpt.get('global_step', 0)
        return ckpt['epoch'], ckpt['loss']
