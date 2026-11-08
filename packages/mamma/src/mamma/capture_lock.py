"""One process-wide lock around CUDA-graph capture and device-wide syncs.

Capturing a CUDA graph on one thread makes any device-wide synchronize on
another thread fail ("operation not permitted when stream is capturing"),
whatever ``capture_error_mode`` says: ultralytics' YOLO calls
``torch.cuda.synchronize()`` during the tracker's (re)detect while the fit
worker captures its Adam graph, and the fitter's bootstrap synchronizes while
the main thread may be capturing. The fitter holds this lock while it
bootstraps or builds a graph; the engine holds it around each tick's
track + landmarks work. Each side is uncontended almost always (the fitter
captures a handful of times per run), so it costs microseconds.
"""

from __future__ import annotations

import threading

GRAPH_CAPTURE_LOCK: threading.RLock = threading.RLock()
