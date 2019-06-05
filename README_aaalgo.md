1. Prerequisite

Follow the original documentation to install all dependencies.
In particular, I'm my working spconv commit is
5df97387208fd9f1b1aa01fb60228f9f99a8a766.  New commits might work.

In addition to that, install the AAA version of Open3D.
https://github.com/aaalgo/Open3D.  The package has been modified to read
the intensity data from PCD files.

My testing environment is:
- Ubuntu 16.04
- python3.6
- pytorch 1.0.0
- cuda 9.0
- TITAN XP -- batch size should be changed if card of different memory
  size is used.

2. Preprocessing

```
python3.6 create_data.py appolo_data_prep
```

3. Train
```
python3.6 ./pytorch/train.py train \
    --config_path=./configs/all.appolo.config \
    --model_dir=models 
```

4. Inference
```
python3.6 ./pytorch/train.py inference
    --config_path=./configs/all.appolo.config
    --model_dir=models
    --info_path=../../data/kitti_infos_test.pkl
    --result_path=./submit
```


5. Training and Testing Data Preparation

Assuming we are in the `second.pytorch/second` directory, all
data should be in `../../data/{train, test}`.

```
second.pytorch/second$ tree ../../data/train/ | head
../../data/train/
├── result_9048_1_frame
│   ├── 233.pcd
│   ├── 238.pcd
│   ├── 243.pcd
│   ├── 248.pcd
│   ├── 253.pcd
│   ├── 258.pcd
│   ├── 263.pcd
│   ├── 268.pcd

second.pytorch/second$ tree ../../data/test/ | head
../../data/test/
├── result_9048_2_frame
│   ├── 102.pcd
│   ├── 107.pcd
│   ├── 112.pcd
│   ├── 117.pcd
│   ├── 122.pcd
│   ├── 127.pcd
│   ├── 12.pcd
│   ├── 132.pcd

```
The content of the PCD zip files should be unpacked and merged into the
`train` and `test` directories.
