import os
import subprocess
import sys
import pytest

NIGHT_VIDEO = os.path.join(os.path.dirname(__file__), "fixtures", "test-video-night.mp4")


def get_gpu_python():
    """Return path to python binary inside venv-gpu if available, else sys.executable."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    win_py = os.path.join(root_dir, "venv-gpu", "Scripts", "python.exe")
    nix_py = os.path.join(root_dir, "venv-gpu", "bin", "python")
    if os.path.exists(win_py):
        return win_py
    if os.path.exists(nix_py):
        return nix_py
    return sys.executable


def is_cuda_available_in_venv(python_bin):
    """Check if CUDA is available in the given python environment."""
    try:
        res = subprocess.run(
            [python_bin, "-c", "import torch; print(torch.cuda.is_available())"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return res.stdout.strip() == "True"
    except Exception:
        return False


def test_video_analyzer_night_fixture(tmp_path):
    python_bin = get_gpu_python()
    if not is_cuda_available_in_venv(python_bin):
        pytest.skip(reason="CUDA GPU is not available in venv-gpu")

    screenshots_dir = tmp_path / "screenshots"
    cfg_file = tmp_path / "override.yaml"
    cfg_file.write_text(f"screenshots:\n  dir: {screenshots_dir.as_posix()}\n", encoding="utf-8")

    res = subprocess.run(
        [python_bin, "video-analyzer.py", NIGHT_VIDEO, "--no-display", "--config", str(cfg_file)],
        capture_output=True,
        text=True,
    )

    assert res.returncode == 0, f"video-analyzer.py failed with stderr:\n{res.stderr}"

    saved_screenshots = list(screenshots_dir.glob("*.jpg"))
    assert len(saved_screenshots) > 0, "Expected screenshot to be saved in test screenshots directory"
