"""
Animation utilities for damage evolution visualization.

Creates GIF and video animations from damage evolution frames.
"""

import os
from pathlib import Path
from typing import List, Optional, Union
import glob
import logging

import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)


def create_damage_evolution_gif(
    frames_dir: Union[str, Path],
    output_path: Union[str, Path],
    fps: int = 5,
    pattern: str = "*_frame_*.png",
    loop: int = 0,
    max_width: int = 1200,
) -> bool:
    """
    Create GIF animation from damage evolution frames.
    
    Args:
        frames_dir: Directory containing frame images
        output_path: Path to save the GIF
        fps: Frames per second
        pattern: Glob pattern for frame files
        loop: Number of loops (0 = infinite)
        max_width: Maximum frame width in pixels for GIF (downscales large frames)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from PIL import Image
    except ImportError:
        logger.error("Pillow not installed. Install with: pip install Pillow")
        return False
    
    frames_dir = Path(frames_dir)
    output_path = Path(output_path)
    
    # Find all frame images
    frame_files = sorted(glob.glob(str(frames_dir / pattern)))
    
    if len(frame_files) == 0:
        # Try alternative pattern
        frame_files = sorted(glob.glob(str(frames_dir / "*.png")))
    
    if len(frame_files) == 0:
        logger.warning(f"No frames found in: {frames_dir}")
        return False
    
    logger.info(f"Creating GIF from {len(frame_files)} frames...")
    
    # Load frames and convert to palette mode with proper dithering.
    # GIF is limited to 256 colours per frame.  Without careful
    # quantization, scientific colour-maps (viridis, plasma) get
    # severely degraded.  We use per-frame adaptive palette with
    # Floyd-Steinberg dithering to preserve smooth gradients.
    frames = []
    for frame_path in tqdm(frame_files, desc="Loading frames"):
        img = Image.open(frame_path)
        # Convert to RGBA first (handles any input mode)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        # Downscale large frames to keep GIF size manageable
        if max_width and img.width > max_width:
            ratio = max_width / img.width
            new_size = (max_width, int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        # Quantize to 256-colour palette with dithering.
        # MEDIANCUT does not support RGBA; FASTOCTREE (method 2) does.
        img_p = img.quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.FLOYDSTEINBERG)
        frames.append(img_p)
    
    if len(frames) == 0:
        logger.warning("No valid frames loaded")
        return False
    
    # Calculate duration per frame in milliseconds
    duration = int(1000 / fps)
    
    # Save GIF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
        optimize=False  # palette already optimised per-frame
    )
    
    logger.info(f"✓ GIF saved to: {output_path}")
    logger.info(f"  - Frames: {len(frames)}")
    logger.info(f"  - Duration: {len(frames) / fps:.1f}s at {fps} fps")
    
    return True


def create_damage_evolution_video(
    frames_dir: Union[str, Path],
    output_path: Union[str, Path],
    fps: int = 10,
    pattern: str = "*_frame_*.png",
) -> bool:
    """
    Create MP4 video from damage evolution frames.
    
    Tries multiple codecs for broad compatibility across different systems.
    Uses frame dimension padding to ensure divisibility by 2 (required by most codecs).
    
    Args:
        frames_dir: Directory containing frame images
        output_path: Path to save the video
        fps: Frames per second
        pattern: Glob pattern for frame files
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV not installed. Install with: pip install opencv-python")
        return False
    
    frames_dir = Path(frames_dir)
    output_path = Path(output_path)
    
    # Ensure output has .mp4 extension
    if output_path.suffix != '.mp4':
        output_path = output_path.with_suffix('.mp4')
    
    # Find all frame images
    frame_files = sorted(glob.glob(str(frames_dir / pattern)))
    
    if len(frame_files) == 0:
        frame_files = sorted(glob.glob(str(frames_dir / "*.png")))
    
    if len(frame_files) == 0:
        logger.warning(f"No frames found in: {frames_dir}")
        return False
    
    logger.info(f"Creating video from {len(frame_files)} frames...")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        logger.error(f"Could not read first frame: {frame_files[0]}")
        return False
    
    height, width = first_frame.shape[:2]
    
    # Ensure dimensions are divisible by 2 (required by most video codecs)
    new_height = height if height % 2 == 0 else height + 1
    new_width = width if width % 2 == 0 else width + 1
    
    if new_height != height or new_width != width:
        logger.info(f"  Padding frame size from {width}x{height} to {new_width}x{new_height} for codec compatibility")
    else:
        logger.info(f"  Frame size: {width}x{height}")
    
    # Create video writer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try multiple codecs in order of preference
    # mp4v is the most portable, XVID works on most systems
    codecs_to_try = [
        ('mp4v', '.mp4'),    # MPEG-4 - most portable
        ('XVID', '.avi'),    # Xvid - widely supported
        ('MJPG', '.avi'),    # Motion JPEG - universal fallback
    ]
    
    video = None
    used_path = output_path
    
    for codec, ext in codecs_to_try:
        try:
            if ext != output_path.suffix:
                used_path = output_path.with_suffix(ext)
            else:
                used_path = output_path
                
            fourcc = cv2.VideoWriter_fourcc(*codec)
            video = cv2.VideoWriter(str(used_path), fourcc, fps, (new_width, new_height), isColor=True)
            
            if video.isOpened():
                logger.info(f"  Using codec: {codec}")
                break
            else:
                video.release()
                video = None
        except Exception as e:
            logger.debug(f"Codec {codec} failed: {e}")
            continue
    
    if video is None or not video.isOpened():
        logger.error("Failed to open video writer with any codec")
        return False
    
    logger.info(f"  Encoding {len(frame_files)} frames at {fps} fps...")
    
    # Write frames
    frames_written = 0
    for frame_path in tqdm(frame_files, desc="Encoding"):
        img = cv2.imread(frame_path)
        if img is not None:
            # Resize/pad frame if necessary for codec compatibility
            if img.shape[0] != new_height or img.shape[1] != new_width:
                # Create padded canvas
                padded = np.zeros((new_height, new_width, 3), dtype=np.uint8)
                # Crop image if larger than target, otherwise just copy
                h_copy = min(img.shape[0], new_height)
                w_copy = min(img.shape[1], new_width)
                padded[:h_copy, :w_copy] = img[:h_copy, :w_copy]
                img = padded
            video.write(img)
            frames_written += 1
    
    video.release()
    
    if frames_written == 0:
        logger.error("No frames were written to video")
        return False
    
    logger.info(f"Video saved to: {used_path} ({frames_written} frames, {frames_written/fps:.1f}s)")
    logger.info(f"  - Frames: {len(frame_files)}")
    logger.info(f"  - Duration: {len(frame_files) / fps:.1f}s at {fps} fps")
    
    return True


def create_animation_from_arrays(
    predictions: List[np.ndarray],
    ground_truth: Optional[List[np.ndarray]],
    mesh_positions: np.ndarray,
    output_dir: Union[str, Path],
    output_name: str = "damage_evolution",
    fps: int = 5,
    times: Optional[List[float]] = None,
    create_gif: bool = True,
    create_video: bool = True,
    data_template = None,
    predicted_rates: Optional[np.ndarray] = None,
    ground_truth_rates: Optional[np.ndarray] = None,
    field_name: str = "Damage",
) -> dict:
    """
    Create animations directly from prediction and ground truth arrays.
    
    This is the main function for testing visualization.
    
    Args:
        predictions: List of predicted damage arrays [num_timesteps, N]
        ground_truth: List of ground truth damage arrays [num_timesteps, N], or None for extended rollout
        mesh_positions: Node positions [N, 3]
        output_dir: Directory to save outputs
        output_name: Base name for output files
        fps: Frames per second
        times: Optional list of time values
        create_gif: Whether to create GIF
        create_video: Whether to create video
        data_template: Optional data object with mesh_pos attribute
        predicted_rates: Actual model-predicted rates [num_timesteps, N] (if None, falls back to damage deltas)
        ground_truth_rates: Ground truth rates [num_timesteps, N] (if None, falls back to damage deltas)
        
    Returns:
        Dictionary with paths to created files
    """
    from .plotting import (
        plot_damage_evolution_frame,
        _is_element_level_data, _build_element_hex_mesh,
        plot_element_damage_frame_perply,
    )

    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    num_timesteps = len(predictions)
    has_ground_truth = ground_truth is not None and len(ground_truth) > 0
    logger.info(f"Creating {num_timesteps} frames (ground truth: {has_ground_truth})...")
    
    # Create a simple data object if not provided
    class SimpleData:
        def __init__(self, pos):
            self.mesh_pos = pos
    
    if data_template is None:
        data_template = SimpleData(mesh_positions)

    # ── Element-level hex mesh (built once, reused for all frames) ──
    hex_mesh = None
    if _is_element_level_data(data_template):
        hex_mesh = _build_element_hex_mesh(data_template)
        if hex_mesh is not None:
            n_iface = len(hex_mesh.get('interfaces', []))
            msg = f"  Using hex-element visualization ({len(hex_mesh['plies'])} plies"
            if n_iface:
                msg += f", {n_iface} interfaces"
            msg += ")"
            logger.info(msg)

    # Generate frames
    for t in tqdm(range(num_timesteps), desc="Generating frames"):
        pred_damage = predictions[t]
        true_damage = ground_truth[t] if has_ground_truth else None

        # Use the model-predicted rates if provided; otherwise fall back
        # to a finite difference of the accumulated damage
        if predicted_rates is not None and t < len(predicted_rates):
            pred_rate = predicted_rates[t]
        elif t > 0:
            pred_rate = predictions[t] - predictions[t-1]
        else:
            pred_rate = np.zeros_like(pred_damage)

        if ground_truth_rates is not None and t < len(ground_truth_rates):
            true_rate = ground_truth_rates[t]
        elif has_ground_truth and t > 0:
            true_rate = ground_truth[t] - ground_truth[t-1]
        else:
            true_rate = np.zeros_like(pred_damage) if has_ground_truth else None

        current_time = times[t] if times is not None else None

        frame_path = frames_dir / f"{output_name}_frame_{t:04d}.png"

        if hex_mesh is not None:
            plot_element_damage_frame_perply(
                data=data_template,
                hex_mesh=hex_mesh,
                pred_damage=pred_damage,
                true_damage=true_damage,
                pred_rate=pred_rate,
                true_rate=true_rate,
                timestep=t,
                save_path=frame_path,
                current_time=current_time,
                field_name=field_name,
            )
        else:
            plot_damage_evolution_frame(
                data=data_template,
                pred_damage=pred_damage,
                true_damage=true_damage,
                pred_rate=pred_rate,
                true_rate=true_rate,
                timestep=t,
                save_path=frame_path,
                current_time=current_time,
                field_name=field_name,
            )
    
    result = {'frames_dir': str(frames_dir)}
    
    # Create GIF
    if create_gif:
        gif_path = output_dir / f"{output_name}.gif"
        if create_damage_evolution_gif(frames_dir, gif_path, fps=fps, 
                                       pattern=f"{output_name}_frame_*.png"):
            result['gif'] = str(gif_path)
    
    # Create video
    if create_video:
        video_path = output_dir / f"{output_name}.mp4"
        if create_damage_evolution_video(frames_dir, video_path, fps=fps,
                                         pattern=f"{output_name}_frame_*.png"):
            result['video'] = str(video_path)

    return result


def create_multi_field_animations(
    rollout_results: dict,
    sim_id: int,
    frame_indices: np.ndarray,
    data_template,
    output_dir: Path,
    fps: int = 5,
    is_two_stage: bool = False,
) -> dict:
    """Generate per-ply element-based GIFs for all field types.

    For each simulation, creates up to 4 sub-directories:
        fi/:          accumulated FI (pred vs GT)
        fi_rate/:     FI rate (pred vs GT)
        damage/:      matrix damage (pred vs GT, two-stage only)
        damage_rate/: matrix damage rate (pred vs GT, two-stage only)

    Each sub-directory contains ``frames/`` PNGs and a ``.gif``.

    Parameters
    ----------
    rollout_results : dict from evaluator with predicted_damage, etc.
    sim_id : int, simulation index
    frame_indices : array of int, indices into rollout arrays for this sim
    data_template : PyG Data for mesh geometry
    output_dir : Path, e.g. ``test_results/damage_evolution/sim_006``
    fps : GIF frame rate
    is_two_stage : whether matrix damage arrays are present

    Returns
    -------
    dict mapping group name -> GIF path (if created)
    """
    from .plotting import (
        _is_element_level_data, _build_element_hex_mesh,
        plot_element_field_perply,
    )

    output_dir = Path(output_dir)
    idx = np.array(frame_indices)

    times = rollout_results.get('times')
    sim_times = times[idx] if times is not None else None

    # Helper: select frames `idx` from a rollout array and return a clean
    # 2-D float32 stack. When the dataset is LHS-sampled (variable node count
    # per plate), the saved rollout arrays are object-dtype and
    # ``arr[idx]`` returns an object array of per-frame arrays which
    # ``np.asarray(_, dtype=float32)`` cannot squash into a single ndarray.
    # Within ONE simulation the node count is constant, so stacking works.
    def _frames(arr):
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.dtype == object:
            return np.stack([np.asarray(arr[i], dtype=np.float32) for i in idx])
        return arr[idx].astype(np.float32, copy=False)

    # Build hex mesh once
    hex_mesh = None
    if _is_element_level_data(data_template):
        hex_mesh = _build_element_hex_mesh(data_template)

    if hex_mesh is None:
        logger.warning("  No hex mesh available; skipping multi-field animations")
        return {}

    # ── Collect arrays for each field group ──
    groups = {}

    # 1. Accumulated FI
    pred_fi = _frames(rollout_results['predicted_damage'])
    gt_fi = _frames(rollout_results['ground_truth_damage'])
    groups['fi'] = {
        'pred': pred_fi, 'gt': gt_fi,
        'label': 'FI', 'cmap': 'viridis', 'rate_norm': False,
    }

    # 1b. Clamped FI (capped at 1 for readability)
    groups['fi_clamped'] = {
        'pred': np.clip(pred_fi, 0, 1), 'gt': np.clip(gt_fi, 0, 1),
        'label': 'FI (clamped)', 'cmap': 'viridis', 'rate_norm': False,
        'vmax': 1.0,
    }

    # 1c. Absolute FI error |pred - gt| (single-field diagnostic).
    # Rendered with no GT so each layer shows one panel; rate_norm keeps
    # small-scale and shock-scale errors visible on the same colorbar.
    groups['fi_error'] = {
        'pred': np.abs(pred_fi - gt_fi), 'gt': None,
        'label': '|pred - GT| FI', 'cmap': 'magma', 'rate_norm': True,
    }

    # 2. FI rate
    pred_fi_rate = _frames(rollout_results.get('predicted_rates'))
    gt_fi_rate = _frames(rollout_results.get('ground_truth_rates'))
    if pred_fi_rate is not None:
        groups['fi_rate'] = {
            'pred': pred_fi_rate, 'gt': gt_fi_rate,
            'label': 'FI rate [1/s]', 'cmap': 'plasma', 'rate_norm': True,
        }

    # 3. Matrix damage (two-stage only)
    pred_mat = rollout_results.get('predicted_matrix_damage')
    gt_mat = rollout_results.get('ground_truth_matrix_damage')
    if is_two_stage and pred_mat is not None and gt_mat is not None:
        pred_mat_sim = _frames(pred_mat)
        gt_mat_sim = _frames(gt_mat)
        groups['damage'] = {
            'pred': pred_mat_sim, 'gt': gt_mat_sim,
            'label': 'matrix damage', 'cmap': 'inferno', 'rate_norm': False,
        }

        # 4. Matrix damage rate (finite difference from accumulated damage)
        pred_dmg_rate = np.zeros_like(pred_mat_sim)
        gt_dmg_rate = np.zeros_like(gt_mat_sim)
        if sim_times is not None and len(sim_times) > 1:
            for ti in range(1, len(pred_mat_sim)):
                dt = sim_times[ti] - sim_times[ti - 1]
                if dt > 0:
                    pred_dmg_rate[ti] = (pred_mat_sim[ti] - pred_mat_sim[ti - 1]) / dt
                    gt_dmg_rate[ti] = (gt_mat_sim[ti] - gt_mat_sim[ti - 1]) / dt
        groups['damage_rate'] = {
            'pred': pred_dmg_rate, 'gt': gt_dmg_rate,
            'label': 'damage rate [1/s]', 'cmap': 'plasma', 'rate_norm': True,
        }

    # ── Optional field-group selection (GTP_ANIM_FIELDS) ──────────────────
    # Unset: render every group. Set to a comma- or space-separated list of
    # group names to render only those, e.g. GTP_ANIM_FIELDS="damage,fi_clamped".
    # An unknown name raises an error.
    _want = os.environ.get("GTP_ANIM_FIELDS", "").replace(",", " ").split()
    if _want:
        _unknown = [g for g in _want if g not in groups]
        if _unknown:
            raise ValueError(
                f"GTP_ANIM_FIELDS: unknown field group(s) {_unknown}. "
                f"Available for this run: {sorted(groups)}")
        _dropped = [g for g in groups if g not in _want]
        groups = {g: c for g, c in groups.items() if g in _want}
        logger.info("  [GTP_ANIM_FIELDS] rendering %s; skipping %s",
                    sorted(groups), _dropped)
        print(f"  [GTP_ANIM_FIELDS] rendering {sorted(groups)}; "
              f"skipping {_dropped}")

    # ── Render frames + GIF for each group ──
    result = {}
    n_frames = len(idx)

    for group_name, cfg in groups.items():
        group_dir = output_dir / group_name
        frames_dir = group_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        pred_arr = cfg['pred']
        gt_arr = cfg['gt']
        has_gt = gt_arr is not None

        for ti in range(n_frames):
            t_val = float(sim_times[ti]) if sim_times is not None else None
            frame_path = frames_dir / f"sim_{sim_id:03d}_{group_name}_frame_{ti:04d}.png"

            plot_element_field_perply(
                hex_mesh=hex_mesh,
                pred_field=pred_arr[ti],
                true_field=gt_arr[ti] if has_gt else None,
                save_path=frame_path,
                field_label=cfg['label'],
                cmap=cfg['cmap'],
                current_time=t_val,
                timestep=ti,
                use_rate_norm=cfg['rate_norm'],
                vmax=cfg.get('vmax'),
            )

        # Create GIF; max_width=None keeps the native resolution of the
        # multi-ply frames
        gif_name = f"sim_{sim_id:03d}_{group_name}.gif"
        gif_path = group_dir / gif_name
        pattern = f"sim_{sim_id:03d}_{group_name}_frame_*.png"
        if create_damage_evolution_gif(frames_dir, gif_path, fps=fps,
                                        pattern=pattern, max_width=None):
            result[group_name] = str(gif_path)

    return result
