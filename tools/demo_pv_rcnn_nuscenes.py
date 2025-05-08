import argparse
import json
from pathlib import Path
import open3d as o3d
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='PV-RCNN NuScenes Demo')
    parser.add_argument('--cfg_file', type=str, required=True, help="Path to config file")
    parser.add_argument('--data_path', type=str, required=True, help="Root path to NuScenes dataset")
    parser.add_argument('--ckpt', type=str, required=True, help="Path to checkpoint")
    parser.add_argument('--output_dir', type=str, default='./output_pvrcnn_nuscenes', help="Output directory")
    parser.add_argument('--num_samples', type=int, default=5, help="Number of samples to process")

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def save_pointcloud_and_boxes(points, boxes, scores, labels, token, output_prefix):
    # Save point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    o3d.io.write_point_cloud(f"{output_prefix}.ply", pcd)

    # Save predictions
    pred_data = {
        "token": token,
        "boxes": boxes.tolist(),
        "scores": scores.tolist(),
        "labels": labels.tolist()
    }

    with open(f"{output_prefix}_pred.json", 'w') as f:
        json.dump(pred_data, f, indent=2)


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('----------------- PV-RCNN NuScenes LiDAR-Only Demo -------------------------')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize dataset
    dataset = NuScenesDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=Path(args.data_path),
        logger=logger
    )

    dataset.infos = dataset.infos[:args.num_samples]
    logger.info(f"Running inference on {len(dataset.infos)} samples.")

    # Load model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for idx, data_dict in enumerate(dataset):
            logger.info(f"Processing sample {idx + 1}/{len(dataset)}")
            data_dict = dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)

            pred_dicts, _ = model.forward(data_dict)

            token = data_dict["metadata"][0]["token"]
            output_prefix = output_dir / f"sample_{idx:03d}"

            save_pointcloud_and_boxes(
                points=data_dict['points'][:, 1:].cpu().numpy(),  # Remove batch idx
                boxes=pred_dicts[0]['pred_boxes'].cpu().numpy(),
                scores=pred_dicts[0]['pred_scores'].cpu().numpy(),
                labels=pred_dicts[0]['pred_labels'].cpu().numpy(),
                token=token,
                output_prefix=str(output_prefix)
            )

            logger.info(f"Saved: {output_prefix}.ply and {output_prefix}_pred.json")

    logger.info('Done saving all outputs.')


if __name__ == '__main__':
    main()
