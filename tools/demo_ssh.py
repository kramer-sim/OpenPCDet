import argparse
import glob
import json
from pathlib import Path

import open3d as o3d
from visual_utils import open3d_vis_utils as V

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--ext', type=str, default='.bin')
    parser.add_argument('--output_dir', type=str, default='/home/berademirhan/OpenPCDet/results')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def save_pointcloud_and_boxes(points, boxes, scores, labels, output_prefix):
    # Save point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    o3d.io.write_point_cloud(f"{output_prefix}.ply", pcd)

    # Save predictions
    pred_data = {
        "boxes": boxes.tolist(),
        "scores": scores.tolist(),
        "labels": labels.tolist()
    }

    with open(f"{output_prefix}_pred.json", 'w') as f:
        json.dump(pred_data, f, indent=2)


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Headless Demo Saving 3D Data-------------------------')

    # Determine subfolder name like "velodyne_000008"
    input_path = Path(args.data_path)
    if input_path.is_file():
        parent_name = input_path.parent.name
        file_stem = input_path.stem
        folder_name = f"{parent_name}_{file_stem}"
    else:
        folder_name = input_path.name

    # Final output directory
    output_dir = Path(args.output_dir) / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=input_path, ext=args.ext, logger=logger
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            output_prefix = output_dir / f"sample_{idx:03d}"
            save_pointcloud_and_boxes(
                points=data_dict['points'][:, 1:].cpu().numpy(),
                boxes=pred_dicts[0]['pred_boxes'].cpu().numpy(),
                scores=pred_dicts[0]['pred_scores'].cpu().numpy(),
                labels=pred_dicts[0]['pred_labels'].cpu().numpy(),
                output_prefix=str(output_prefix)
            )

            logger.info(f"Saved: {output_prefix}.ply and {output_prefix}_pred.json")

    logger.info('Done saving all outputs.')


if __name__ == '__main__':
    main()
