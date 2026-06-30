"""
Render ground-truth synthetic shadows next to shadows re-rendered from a
predicted light position.

The trained light predictor outputs a 3D light position, not a shadow mask.
This script keeps each generated scene's objects and camera fixed, predicts the
light position from the ground-truth shadow image, then re-renders the scene
with the predicted light.
"""

import argparse
import glob
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import torch

from shadow_prediction.model import create_model
from utils.synthetic_data import (
    compute_shadow,
    generate_synthetic_scene,
    rasterize_convex_hull,
)


def _projection_basis(scene_data):
    normal = scene_data["camera_pos"] / np.linalg.norm(scene_data["camera_pos"])

    u_vec = np.array([-normal[1], normal[0], 0.0])
    u_vec = u_vec / np.linalg.norm(u_vec)

    v_vec = np.cross(normal, u_vec)
    v_vec = v_vec / np.linalg.norm(v_vec)
    if v_vec[2] < 0:
        v_vec = -v_vec

    return u_vec, v_vec


def _project_to_plane(x, y, z, u_vec, v_vec):
    points_3d = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    u = np.dot(points_3d, u_vec)
    v = np.dot(points_3d, v_vec)
    return u, v


def render_scene_with_light(scene_data, light_pos, img_width=224, img_height=224):
    """Render the original scene geometry with a supplied light position."""
    u_vec, v_vec = _projection_basis(scene_data)

    cyl_shadow_x, cyl_shadow_y = compute_shadow(
        scene_data["cyl_x"], scene_data["cyl_y"], scene_data["cyl_z"], light_pos
    )
    cone_shadow_x, cone_shadow_y = compute_shadow(
        scene_data["cone_x"], scene_data["cone_y"], scene_data["cone_z"], light_pos
    )

    cyl_shadow_u, cyl_shadow_v = _project_to_plane(
        cyl_shadow_x, cyl_shadow_y, np.zeros_like(cyl_shadow_x), u_vec, v_vec
    )
    cone_shadow_u, cone_shadow_v = _project_to_plane(
        cone_shadow_x, cone_shadow_y, np.zeros_like(cone_shadow_x), u_vec, v_vec
    )

    cyl_shadow_points = np.column_stack([cyl_shadow_u, cyl_shadow_v])
    cone_shadow_points = np.column_stack([cone_shadow_u, cone_shadow_v])

    image = np.ones((img_height, img_width))

    if len(cyl_shadow_points) > 3:
        image = rasterize_convex_hull(
            image,
            cyl_shadow_points,
            scene_data["u_range"],
            scene_data["v_range"],
            img_width,
            img_height,
            0.5,
        )
    if len(cone_shadow_points) > 3:
        image = rasterize_convex_hull(
            image,
            cone_shadow_points,
            scene_data["u_range"],
            scene_data["v_range"],
            img_width,
            img_height,
            0.5,
        )

    image = rasterize_convex_hull(
        image,
        scene_data["cyl_points"],
        scene_data["u_range"],
        scene_data["v_range"],
        img_width,
        img_height,
        0,
    )
    image = rasterize_convex_hull(
        image,
        scene_data["cone_points"],
        scene_data["u_range"],
        scene_data["v_range"],
        img_width,
        img_height,
        0,
    )

    return image


def load_model(checkpoint_path, device):
    model = create_model(pretrained=False, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def resolve_checkpoint(checkpoint_path):
    if checkpoint_path:
        return checkpoint_path

    candidates = glob.glob("models_checkpoints/*/checkpoint_best.pth")
    if not candidates:
        raise FileNotFoundError(
            "No checkpoint found. Train a model first or pass --checkpoint PATH."
        )

    return max(candidates, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--output",
        default="assets/shadow_prediction_examples.png",
    )
    parser.add_argument("--num-scenes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    model = load_model(checkpoint_path, device)

    fig, axes = plt.subplots(
        args.num_scenes,
        2,
        figsize=(7, 3.2 * args.num_scenes),
        constrained_layout=True,
    )
    if args.num_scenes == 1:
        axes = np.expand_dims(axes, axis=0)

    errors = []

    for idx in range(args.num_scenes):
        real_image, scene_data = generate_synthetic_scene(
            random_projection=False,
            randomize_objects=True,
            add_shadows=True,
            img_width=args.img_size,
            img_height=args.img_size,
            return_scene_data=True,
        )

        model_input = torch.from_numpy(real_image).float().unsqueeze(0).to(device)
        with torch.no_grad():
            pred_light = model(model_input).squeeze(0).cpu().numpy()

        true_light = scene_data["light_pos"]
        light_error = float(np.linalg.norm(pred_light - true_light))
        errors.append(light_error)

        pred_image = render_scene_with_light(
            scene_data,
            pred_light,
            img_width=args.img_size,
            img_height=args.img_size,
        )

        row_axes = axes[idx]
        row_axes[0].imshow(real_image.squeeze(0), cmap="gray", vmin=0, vmax=1)
        row_axes[0].set_title(
            f"Scene {idx + 1}: ground truth\n"
            f"light=({true_light[0]:.1f}, {true_light[1]:.1f}, {true_light[2]:.1f})"
        )
        row_axes[1].imshow(pred_image, cmap="gray", vmin=0, vmax=1)
        row_axes[1].set_title(
            "Predicted light re-render\n"
            f"pred=({pred_light[0]:.1f}, {pred_light[1]:.1f}, {pred_light[2]:.1f}), "
            f"err={light_error:.2f}"
        )

        for ax in row_axes:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"Ground-truth shadows vs re-rendered shadows from predicted light "
        f"(mean light error={np.mean(errors):.2f})",
        fontsize=14,
    )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Saved comparison to {args.output}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Mean light error: {np.mean(errors):.4f}")
    print(f"Median light error: {np.median(errors):.4f}")


if __name__ == "__main__":
    main()
