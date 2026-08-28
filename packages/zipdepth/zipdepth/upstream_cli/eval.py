"""
ZipDepth zero-shot depth evaluation.

Usage:
    python scripts/eval.py --dataset nyuv2 \
        --data_dir /path/to/NYUv2/test \
        --checkpoint checkpoints/zipdepth_base.pth

Supported datasets: nyuv2, kitti, eth3d, scannet, diode
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from monopriors.models.relative_depth.zipdepth import ZipDepthPredictor

from zipdepth.evaluation import DATASET_CONFIGS, discover_samples, evaluate
from zipdepth.evaluation.evaluator import print_results, save_results

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class _EvaluatorAdapter:
    """``evaluate`` expects ``infer_image(bgr) -> inverse depth``; monopriors' predictor takes RGB."""

    def __init__(self, predictor: ZipDepthPredictor) -> None:
        self.predictor = predictor

    def infer_image(self, bgr: np.ndarray) -> np.ndarray:
        return self.predictor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), None).disparity


def main():
    parser = argparse.ArgumentParser(description='ZipDepth zero-shot evaluation')

    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Benchmark to evaluate on')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset root')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the ZipDepth checkpoint (.pth)')
    parser.add_argument('--variant', type=str, default='base',
                        choices=['small', 'base', 'large', 'giant'])
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                        help='Output directory (a per-dataset subfolder is created)')

    # Inference options
    parser.add_argument('--input-size', type=int, default=384,
                        help='Shorter-side size for model input')
    parser.add_argument('--npu', action='store_true',
                        help='NPU-compatible upsampling (required for *_npu.pth)')
    parser.add_argument('--fp16', action='store_true',
                        help='FP16 inference (CUDA only)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])

    # Protocol options
    parser.add_argument('--alignment', type=str, default='least_square_disparity',
                        choices=['least_square', 'least_square_disparity'],
                        help='Scale/shift alignment space')
    parser.add_argument('--alignment_max_res', type=int, default=None,
                        help='Downsample resolution for the least-squares solve')
    parser.add_argument('--no_kitti_crop', action='store_true',
                        help='Disable the KITTI benchmark crop')
    parser.add_argument('--eval_mask', type=str, default=None,
                        choices=['garg', 'eigen', 'none'],
                        help='Override the dataset evaluation mask')
    parser.add_argument('--diode-domain', type=str, default='all',
                        choices=['indoors', 'outdoor', 'all'],
                        help='DIODE subset to evaluate')

    # Misc
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit the number of samples (debugging)')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save raw predictions as .npy')

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA not available — falling back to CPU")
        args.device = 'cpu'

    cfg = dict(DATASET_CONFIGS[args.dataset])
    if args.dataset == 'diode':
        cfg['domain'] = args.diode_domain

    samples = discover_samples(args.dataset, args.data_dir, domain=args.diode_domain)
    if not samples:
        logging.error("No samples found — check --data_dir and the dataset layout.")
        sys.exit(1)
    if args.max_samples:
        samples = samples[:args.max_samples]

    predictor = _EvaluatorAdapter(ZipDepthPredictor(
        device=args.device, checkpoint=Path(args.checkpoint), input_size=args.input_size, npu=args.npu,
    ))

    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_mask_type = (None if args.eval_mask == 'none' else args.eval_mask) if args.eval_mask is not None else cfg.get('eval_mask')

    print(f"\n{'='*52}")
    print(f"  ZipDepth evaluation — {args.dataset}")
    print(f"{'='*52}")
    print(f"  Samples:    {len(samples)}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Inference:  {args.device}, {'FP16' if args.fp16 else 'FP32'}"
          f"{', NPU upsampling' if args.npu else ''}")
    print(f"  Alignment:  {args.alignment}")
    print(f"  Depth range:[{cfg['min_depth']}, {cfg['max_depth']}] m")
    print(f"  Output:     {output_dir}")
    print(f"{'='*52}\n")

    results = evaluate(
        predictor=predictor,
        samples=samples,
        cfg=cfg,
        alignment=args.alignment,
        alignment_max_res=args.alignment_max_res,
        apply_kitti_crop=not args.no_kitti_crop,
        eval_mask_type=eval_mask_type,
        output_dir=str(output_dir),
        save_predictions=args.save_predictions,
    )

    print_results(results, args.dataset, len(samples))
    save_results(results, args.dataset, str(output_dir), args.alignment, args.checkpoint)


if __name__ == '__main__':
    main()
