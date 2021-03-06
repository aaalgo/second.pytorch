import copy
from pathlib import Path
import pickle

import fire
import open3d

import second.data.kitti_dataset as kitti_ds
import second.data.nuscenes_dataset as nu_ds
import second.data.appolo_dataset as appolo_ds
from second.data.all_dataset import create_groundtruth_database

def kitti_data_prep(root_path="/scratch1/data/kitti"):
    kitti_ds.create_kitti_info_file(root_path)
    kitti_ds.create_reduced_point_cloud(root_path)
    create_groundtruth_database("KittiDataset", root_path, Path(root_path) / "kitti_infos_train.pkl")

def nuscenes_data_prep(root_path, version, dataset_name, max_sweeps=10):
    nu_ds.create_nuscenes_infos(root_path, version=version, max_sweeps=max_sweeps)
    name = "infos_train.pkl"
    if version == "v1.0-test":
        name = "infos_test.pkl"
    create_groundtruth_database(dataset_name, root_path, Path(root_path) / name)

def appolo_data_prep (root_path="/scratch2/wdong/appolo/data"):
    appolo_ds.create_kitti_info_file(root_path)
    create_groundtruth_database("AppoloDataset", root_path, Path(root_path) / "kitti_infos_train.pkl")
    pass

if __name__ == '__main__':
    fire.Fire()
