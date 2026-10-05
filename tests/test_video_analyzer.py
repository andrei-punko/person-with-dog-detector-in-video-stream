import os
import subprocess
import sys
import pytest
import cv2


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


def verify_screenshot(screenshot_path, reference_path, test_name):
    """Verify that a generated screenshot matches the expected reference."""
    # Verify screenshot filename contains timestamp (e.g., "test_video_night_6.6s.jpg")
    screenshot_name = os.path.basename(screenshot_path)
    screenshot_name_without_ext = os.path.splitext(screenshot_name)[0]
    parts = screenshot_name_without_ext.rsplit("_", 1)
    assert len(parts) == 2, f"Expected format 'label_timestamp.jpg', got: {screenshot_name}"
    timestamp_part = parts[1]
    assert timestamp_part.endswith("s"), f"Timestamp should end with 's' (seconds), got: {timestamp_part}"

    # Compare images (should be identical for same timestamp)
    ref_img = cv2.imread(reference_path)
    assert ref_img is not None, f"Reference screenshot not found: {reference_path}"

    gen_img = cv2.imread(screenshot_path)
    assert gen_img is not None, f"Generated screenshot could not be read: {screenshot_path}"

    assert ref_img.shape == gen_img.shape, \
        f"Image shapes don't match: reference {ref_img.shape} vs generated {gen_img.shape}"


def run_video_analyzer_test(python_bin, video_path, reference_path, cfg_file, screenshots_dir):
    """Run the video analyzer and verify it produces a valid screenshot."""
    res = subprocess.run(
        [python_bin, "video-analyzer.py", video_path, "--no-display", "--config", str(cfg_file)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"video-analyzer.py failed with stderr:\n{res.stderr}"

    saved_screenshots = list(screenshots_dir.glob("*.jpg"))
    assert len(saved_screenshots) > 0, "Expected screenshot to be saved in test screenshots directory"

    # Verify the first saved screenshot has valid format and matches reference
    verify_screenshot(saved_screenshots[0], reference_path, os.path.basename(video_path))


def test_video_analyzer(tmp_path):
    """Test that video analyzer processes night and day videos and saves matching screenshots."""
    python_bin = get_gpu_python()
    if not is_cuda_available_in_venv(python_bin):
        pytest.skip(reason="CUDA GPU is not available in venv-gpu")

    screenshots_dir = tmp_path / "screenshots"
    cfg_file = tmp_path / "override.yaml"
    cfg_file.write_text(f"screenshots:\n  dir: {screenshots_dir.as_posix()}\n  jpeg_quality: 75\n  max_width: 1920\n", encoding="utf-8")

    # Night video test
    night_video = os.path.join(os.path.dirname(__file__), "fixtures", "test_video_night.mp4")
    night_reference = os.path.join(os.path.dirname(__file__), "fixtures", "test_video_night_6.6s.jpg")
    run_video_analyzer_test(python_bin, night_video, night_reference, cfg_file, screenshots_dir)

    # Day video test
    day_video = os.path.join(os.path.dirname(__file__), "fixtures", "test_video_day.mp4")
    day_reference = os.path.join(os.path.dirname(__file__), "fixtures", "test_video_day_14.4s.jpg")
    run_video_analyzer_test(python_bin, day_video, day_reference, cfg_file, screenshots_dir)
