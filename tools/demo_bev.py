import argparse
import json
from pathlib import Path
import os
import imageio
import cv2
import shutil
import open3d as o3d
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pprint import pprint



def parse_config():
    parser = argparse.ArgumentParser(description='BEVFusion NuScenes Demo')
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True, help="Root path of NuScenes dataset")
    parser.add_argument('--ckpt', type=str, required=True, help="Model checkpoint path")
    parser.add_argument('--output_dir', type=str, default='./output', help="Where to save predictions")
    parser.add_argument('--num_samples', type=int, default=5, help="Number of samples to run")
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def save_pointcloud_and_boxes(points, boxes, scores, labels, data_dict, output_prefix):
    # Save point cloud
    token = data_dict["metadata"][0]["token"]
    print("token: ", token)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    o3d.io.write_point_cloud(f"{output_prefix}.ply", pcd)

    # Save predictions including token
    pred_data = {
        "token": token,
        "boxes": boxes.tolist(),
        "scores": scores.tolist(),
        "labels": labels.tolist()
    }

    with open(f"{output_prefix}_pred.json", 'w') as f:
        json.dump(pred_data, f, indent=2)


def save_camera_images(camera_imgs, output_prefix, data_dict, logger, dataset_root):
    # Ensure output directory exists
    os.makedirs(output_prefix, exist_ok=True)


    # Retrieve the image paths from 'data_dict['image_paths']'
    image_paths = data_dict['image_paths'][0]  # First sample (assuming batch size is 1)

    # Now iterate over the camera types and corresponding image paths
    camera_types = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
    for cam_idx, cam_image_path in enumerate(image_paths):
        # Build full path by joining dataset_root and the relative image path
        full_image_path = os.path.join(dataset_root, cam_image_path)
        
        # Prepare output path
        output_image_path = os.path.join(output_prefix, f"{camera_types[cam_idx]}.jpg")
        
        # Log the camera image path
        logger.info(f"Saving {camera_types[cam_idx]} image to: {output_image_path}")
        
        # Copy the image from source to output directory
        if os.path.exists(full_image_path):
            shutil.copy(full_image_path, output_image_path)
        else:
            logger.warning(f"Image not found: {full_image_path}")


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Headless BEVFusion NuScenes Demo-------------------------')

    dataset_root = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize NuScenes dataset
    demo_dataset = NuScenesDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=dataset_root,
        logger=logger
    )

    # Limit number of samples
    demo_dataset.infos = demo_dataset.infos[:args.num_samples]
    logger.info(f"Running inference on {len(demo_dataset.infos)} samples.")

    # Load model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f"Processing sample {idx + 1}/{len(demo_dataset)}")
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)

            # Create output folder for each sample (e.g., sample_000)
            output_prefix = output_dir / f"sample_{idx:03d}"
            save_camera_images(data_dict['camera_imgs'], output_prefix, data_dict, logger, args.data_path)

            # Inference
            pred_dicts, _ = model.forward(data_dict)

            save_pointcloud_and_boxes(
                points=data_dict['points'][:, 1:].cpu().numpy(),  # skip batch index
                boxes=pred_dicts[0]['pred_boxes'].cpu().numpy(),
                scores=pred_dicts[0]['pred_scores'].cpu().numpy(),
                labels=pred_dicts[0]['pred_labels'].cpu().numpy(),
                data_dict=data_dict,
                output_prefix=str(output_prefix)
            )

            logger.info(f"Saved: {output_prefix}.ply and {output_prefix}_pred.json")

    logger.info('Done saving all outputs.')


if __name__ == '__main__':
    main()
