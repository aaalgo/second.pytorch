import os
import sys
from pathlib import Path
import pickle
import time
from functools import partial
import open3d

from random import shuffle
import numpy as np
from glob import glob
import fire


from second.core import box_np_ops
from second.core import preprocess as prep
from second.data import kitti_common as kitti
from second.utils.eval import get_coco_eval_result, get_official_eval_result
from second.data.dataset import Dataset, register_dataset
from second.utils.progress_bar import progress_bar_iter as prog_bar

CLASS_LABELS = ["", "Car", "Van", "Pedestrian", "Cyclist", "DontCare", "DontCare"]

@register_dataset
class AppoloDataset(Dataset):
    NumPointFeatures = 4

    def __init__(self,
                 root_path,
                 info_path,
                 class_names=None,
                 prep_func=None,
                 num_point_features=None):
        assert info_path is not None
        print("LOAD DATASET:", info_path)
        print("CLASS_NAMES", class_names)
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        #self._root_path = Path(root_path)
        self._kitti_infos = infos

        print("remain number of infos:", len(self._kitti_infos))
        self._class_names = class_names
        self._prep_func = prep_func

    def __len__(self):
        return len(self._kitti_infos)

    def convert_detection_to_kitti_annos(self, detection):
        class_names = self._class_names
        det_image_idxes = [det["metadata"]["image_idx"] for det in detection]
        gt_image_idxes = [
            info["image"]["image_idx"] for info in self._kitti_infos
        ]
        annos = []
        for i in range(len(detection)):
            det_idx = det_image_idxes[i]
            det = detection[i]
            # info = self._kitti_infos[gt_image_idxes.index(det_idx)]
            info = self._kitti_infos[i]
            calib = info["calib"]
            rect = calib["R0_rect"]
            Trv2c = calib["Tr_velo_to_cam"]
            P2 = calib["P2"]
            final_box_preds = det["box3d_lidar"].detach().cpu().numpy()
            label_preds = det["label_preds"].detach().cpu().numpy()
            scores = det["scores"].detach().cpu().numpy()
            if final_box_preds.shape[0] != 0:
                final_box_preds[:, 2] -= final_box_preds[:, 5] / 2
                box3d_camera = box_np_ops.box_lidar_to_camera(
                    final_box_preds, rect, Trv2c)
                locs = box3d_camera[:, :3]
                dims = box3d_camera[:, 3:6]
                angles = box3d_camera[:, 6]
                camera_box_origin = [0.5, 1.0, 0.5]
                box_corners = box_np_ops.center_to_corner_box3d(
                    locs, dims, angles, camera_box_origin, axis=1)
                box_corners_in_image = box_np_ops.project_to_image(
                    box_corners, P2)
                # box_corners_in_image: [N, 8, 2]
                minxy = np.min(box_corners_in_image, axis=1)
                maxxy = np.max(box_corners_in_image, axis=1)
                bbox = np.concatenate([minxy, maxxy], axis=1)
            anno = kitti.get_start_result_anno()
            num_example = 0
            box3d_lidar = final_box_preds
            for j in range(box3d_lidar.shape[0]):
                image_shape = info["image"]["image_shape"]
                if bbox[j, 0] > image_shape[1] or bbox[j, 1] > image_shape[0]:
                    continue
                if bbox[j, 2] < 0 or bbox[j, 3] < 0:
                    continue
                bbox[j, 2:] = np.minimum(bbox[j, 2:], image_shape[::-1])
                bbox[j, :2] = np.maximum(bbox[j, :2], [0, 0])
                anno["bbox"].append(bbox[j])
                # convert center format to kitti format
                # box3d_lidar[j, 2] -= box3d_lidar[j, 5] / 2
                anno["alpha"].append(
                    -np.arctan2(-box3d_lidar[j, 1], box3d_lidar[j, 0]) +
                    box3d_camera[j, 6])
                anno["dimensions"].append(box3d_camera[j, 3:6])
                anno["location"].append(box3d_camera[j, :3])
                anno["rotation_y"].append(box3d_camera[j, 6])

                anno["name"].append(class_names[int(label_preds[j])])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["score"].append(scores[j])

                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(kitti.empty_result_anno())
            num_example = annos[-1]["name"].shape[0]
            annos[-1]["metadata"] = det["metadata"]
        return annos

    def evaluation(self, detections, output_dir):
        """
        detection
        When you want to eval your own dataset, you MUST set correct
        the z axis and box z center.
        If you want to eval by my KITTI eval function, you must 
        provide the correct format annotations.
        ground_truth_annotations format:
        {
            bbox: [N, 4], if you fill fake data, MUST HAVE >25 HEIGHT!!!!!!
            alpha: [N], you can use -10 to ignore it.
            occluded: [N], you can use zero.
            truncated: [N], you can use zero.
            name: [N]
            location: [N, 3] center of 3d box.
            dimensions: [N, 3] dim of 3d box.
            rotation_y: [N] angle.
        }
        all fields must be filled, but some fields can fill
        zero.
        """
        if "annos" not in self._kitti_infos[0]:
            return None
        gt_annos = [info["annos"] for info in self._kitti_infos]
        dt_annos = self.convert_detection_to_kitti_annos(detections)
        # firstly convert standard detection to kitti-format dt annos
        z_axis = 1  # KITTI camera format use y as regular "z" axis.
        z_center = 1.0  # KITTI camera box's center is [0.5, 1, 0.5]
        # for regular raw lidar data, z_axis = 2, z_center = 0.5.
        result_official_dict = get_official_eval_result(
            gt_annos,
            dt_annos,
            self._class_names,
            z_axis=z_axis,
            z_center=z_center)
        result_coco = get_coco_eval_result(
            gt_annos,
            dt_annos,
            self._class_names,
            z_axis=z_axis,
            z_center=z_center)
        return {
            "results": {
                "official": result_official_dict["result"],
                "coco": result_coco["result"],
            },
            "detail": {
                "eval.kitti": {
                    "official": result_official_dict["detail"],
                    "coco": result_coco["detail"]
                }
            },
        }

    def __getitem__(self, idx):
        input_dict = self.get_sensor_data(idx)

        example = self._prep_func(input_dict=input_dict)
        example["metadata"] = {'pcd_path': self._kitti_infos[idx]["point_cloud"]['velodyne_path']}
        if "image_idx" in input_dict["metadata"]:
            example["metadata"] = input_dict["metadata"]
        if "anchors_mask" in example:
            example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
        return example

    def get_sensor_data(self, query):
        read_image = False
        idx = query
        if isinstance(query, dict):
            assert False
            '''
            read_image = "cam" in query
            assert "lidar" in query
            idx = query["lidar"]["idx"]
            '''
        info = self._kitti_infos[idx]
        pc_info = info["point_cloud"]
        velo_path = pc_info['velodyne_path']

        res = {
                "metadata": {},
            "lidar": {
                "type": "lidar",
                "points": None,
            },
        }

        #assert velo_path.is_absolute()

        if True:    # load PCD
            print('LOADING', velo_path)
            pcd = open3d.read_point_cloud(velo_path)
            print(pcd)
            pcd_pp = np.asarray(pcd.points)
            pcd_cc = np.asarray(pcd.intensity)[:, np.newaxis]
            points = np.concatenate([pcd_pp, pcd_cc], axis=1)
            points = points.astype(np.float32)
            pass

        '''
        points = np.fromfile(
            str(velo_path), dtype=np.float32,
            count=-1).reshape([-1, self.NumPointFeatures])
        '''
        res["lidar"]["points"] = points
        if 'annos' in info:
            annos = info['annos']
            # we need other objects to avoid collision when sample
            # annos = kitti.remove_dontcare(annos)
            gt_boxes = np.array(annos['boxes'], dtype=np.float32)
            gt_names = np.array(annos["name"])
            # rots = np.concatenate([np.zeros([locs.shape[0], 2], dtype=np.float32), rots], axis=1)
            #gt_boxes = np.concatenate([locs, dims, rots[..., np.newaxis]],
            #                          axis=1).astype(np.float32)
            # only center format is allowed. so we need to convert
            # kitti [0.5, 0.5, 0] center to [0.5, 0.5, 0.5]

            #box_np_ops.change_box3d_center_(gt_boxes, [0.5, 0.5, 0],
            #                                [0.5, 0.5, 0.5])
            assert len(gt_names) == gt_boxes.shape[0]
            res["lidar"]["annotations"] = {
                'boxes': gt_boxes,
                'names': gt_names,
            }

        anno_dict = res["lidar"]["annotations"]
        assert(anno_dict["boxes"].shape[0] == len(anno_dict["names"]))

        return res


def _calculate_num_points_in_gt(data_path,
                                infos,
                                remove_outside=True,
                                num_features=4):
    for info in infos:
        pc_info = info["point_cloud"]
        v_path = pc_info["velodyne_path"]
        if True:
            pcd = open3d.read_point_cloud(v_path)
            points_v = np.asarray(pcd.points).astype(np.float32)

        #points_v = np.fromfile(
        #    v_path, dtype=np.float32, count=-1).reshape([-1, num_features])

        # points_v = points_v[points_v[:, 0] > 0]
        annos = info['annos']
        num_obj = len(annos['name']) #len([n for n in annos['name'] if n != 'DontCare'])
        # annos = kitti.filter_kitti_anno(annos, ['DontCare'])
        gt_boxes = np.array(annos['boxes'], dtype=np.float32)

        #gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]],
        #                                 axis=1)
        #gt_boxes_lidar = box_np_ops.box_camera_to_lidar(
        #    gt_boxes_camera, rect, Trv2c)
        indices = box_np_ops.points_in_rbbox(points_v, gt_boxes)
        num_points_in_gt = indices.sum(0)
        #num_ignored = len(annos['dimensions']) - num_obj
        #num_points_in_gt = np.concatenate(
        #    [num_points_in_gt, -np.ones([num_ignored])])
        annos["num_points_in_gt"] = num_points_in_gt.astype(np.int32)
        pass

def get_appolo_image_info (training, stems):
    infos = []
    for stem in stems:
        frames = {}
        for pcd_path in glob(stem + "_frame/*.pcd"):
            bn = os.path.basename(pcd_path)
            frame = int(bn.split('.')[0])
            info = {
                    "point_cloud": {
                            "velodyne_path": pcd_path
                        },
                    "annos": {
                            "boxes": [],
                            "name": [],
                            "wdong_cnt": 0
                        }
                            
                    }
            frames[frame] = info
            pass
        if training:
            with open(stem + '.txt', 'r') as f:
                for line in f:
                    fs = line.strip().split(' ')
                    assert len(fs) == 9
                    frame = int(fs[0])

                    assert frame in frames
                    info = frames[frame]
                    category = int(fs[1])
                    assert category > 0
                    if category <= 4:
                        info["annos"]["wdong_cnt"] += 1
                    info["annos"]["name"].append(CLASS_LABELS[category])
                    info["annos"]["boxes"].append([float(x) for x in fs[2:]])
                    pass
                pass
            # read labels
            pass
        for k, info in frames.items():
            assert len(info["annos"]["name"]) == len(info["annos"]["boxes"])
            #if info["annos"]["wdong_cnt"] == 0 and training:
            #    continue
            infos.append(info)
            pass
        pass
    print("got %d images" % len(infos))
    return infos


def create_kitti_info_file(data_path, save_path=None):
    train_dirs = [x.replace("_frame", "") for x in glob(data_path + "/train/*_frame")]
    shuffle(train_dirs)
    split = 5
    n_val = len(train_dirs) // split
    print("Loading %d dirs" % len(train_dirs))
    val_dirs = train_dirs[:n_val]
    train_dirs = train_dirs[n_val:]
    test_dirs = [x.replace("_frame", "") for x in glob(data_path + "/test/*_frame")]
    print("# train dirs: %d" % len(train_dirs))
    print("# val dirs: %d" % len(val_dirs))
    print("# test dirs: %d" % len(test_dirs))


    print("Generate info. this may take several minutes.")
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    kitti_infos_test = get_appolo_image_info(False, test_dirs)
    filename = save_path / 'kitti_infos_test.pkl'
    print(f"Kitti info test file (%d) is saved to {filename}" % len(kitti_infos_test))
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)

    kitti_infos_train = get_appolo_image_info(True, train_dirs)
    #_calculate_num_points_in_gt(data_path, kitti_infos_train)
    filename = save_path / 'kitti_infos_train.pkl'
    print(f"Kitti info train file (%d) is saved to {filename}" % len(kitti_infos_train))
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    kitti_infos_val = get_appolo_image_info(True, val_dirs)
    #_calculate_num_points_in_gt(data_path, kitti_infos_val)
    filename = save_path / 'kitti_infos_val.pkl'
    print(f"Kitti info val file (%d) is saved to {filename}" % len(kitti_infos_val))
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    filename = save_path / 'kitti_infos_trainval.pkl'
    print(f"Kitti info trainval file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)

    kitti_infos_test = get_appolo_image_info(False, test_dirs)
    filename = save_path / 'kitti_infos_test.pkl'
    print(f"Kitti info test file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)


if __name__ == "__main__":
    fire.Fire()
