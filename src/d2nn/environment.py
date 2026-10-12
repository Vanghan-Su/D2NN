"""Environment and device diagnostics."""

from __future__ import annotations

import platform
import sys


def environment_summary() -> str:
    lines = [
        f"python={sys.version.split()[0]}",
        f"platform={platform.platform()}",
    ]
    try:
        import numpy as np
        lines.append(f"numpy={np.__version__}")
    except Exception as exc:  # pragma: no cover - diagnostic only
        lines.append(f"numpy=unavailable ({exc})")
    try:
        import torch
        lines.append(f"torch={torch.__version__}")
        lines.append(f"torch_cuda={torch.version.cuda}")
        lines.append(f"cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"gpu={torch.cuda.get_device_name(0)}")
    except Exception as exc:  # pragma: no cover - diagnostic only
        lines.append(f"torch=unavailable ({exc})")
    return "\n".join(lines)
