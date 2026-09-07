"""Shared test setup.

Silence the huggingface_hub symlink warning that fires on Windows machines
without Developer Mode — caching still works, just without symlinks.
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
