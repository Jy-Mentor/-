"""警告过滤工具.

用于抑制已知且暂时无法修复的第三方库弃用警告, 保持终端输出整洁.
"""

from __future__ import annotations

import warnings


def suppress_known_library_warnings() -> None:
    """抑制 torch_geometric / torch 等已知弃用警告.

    这些警告来自第三方库内部实现, 不影响当前功能.
    当库升级后应重新评估是否移除过滤.
    """
    warnings.filterwarnings(
        "ignore",
        message="`torch_geometric.distributed` has been deprecated",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="`torch.jit.script` is deprecated",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Failing to pass a value to the 'type_params' parameter",
        category=DeprecationWarning,
    )
