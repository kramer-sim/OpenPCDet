import argparse
from easydict import EasyDict
from pathlib import Path
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset

def create_nuscenes_infos(version, data_path, max_sweeps=10, with_cam=True):
    # Define configuration here, including the 'src_feature_list'
    dataset_cfg = EasyDict({
        'VERSION': version,
        'DATA_PATH': data_path,
        'MAX_SWEEPS': max_sweeps,
        'USE_CAMERA': with_cam,
        'POINT_CLOUD_RANGE': [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
        'POINT_FEATURE_ENCODING': {
            'num_point_features': 5,
            'encoding': 'raw'
        },
        'src_feature_list': ['x', 'y', 'z', 'intensity', 'ring_index'],  # Make sure this is defined
    })

    # Make sure to use pathlib to join paths correctly
    root_path = Path(data_path) / dataset_cfg.VERSION

    # Now pass this to the NuScenesDataset
    dataset = NuScenesDataset(
        dataset_cfg=dataset_cfg,
        class_names=[],  # Empty or filled with class names if needed
        training=True,
        root_path=root_path,
        logger=None
    )
    dataset.create_nuscenes_infos(with_cam=with_cam)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='v1.0-mini')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--max_sweeps', type=int, default=10)
    parser.add_argument('--with_cam', action='store_true')
    args = parser.parse_args()

    create_nuscenes_infos(
        version=args.version,
        data_path=args.data_path,
        max_sweeps=args.max_sweeps,
        with_cam=args.with_cam
    )
