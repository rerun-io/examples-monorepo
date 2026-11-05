# Copyright (c) Meta Platforms, Inc. and affiliates.
import logging


def get_pylogger(name=__name__) -> logging.Logger:
    """Command line logger (upstream gates it to rank zero for multi-GPU training)."""
    return logging.getLogger(name)
