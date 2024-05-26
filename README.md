# IVP - Infilling Vector Prediction
Official Code Release accompanying the WACV 2022 paper "Revealing Disocclusions in Temporal View Synthesis through Infilling Vector Prediction"

## Databases
* Download the [IISc VEED Database](https://nagabhushansn95.github.io/publications/2021/ivp.html#database-download). Extract the zip files and place them in `Data/Veed`.
* For SceneNet database, download all the ground truth for the training set from [here](https://robotvault.bitbucket.io/scenenet-rgbd.html). Extract the zip files and place them in `Data/SceneNet`. The following steps describe training and testing on IISc VEED dataset. The steps for SceneNet dataset are similar and the code for each step is also provided.

## Python Environment
Environment details are available in `IVP.yml` for `conda` and `requirements.txt` for pip. To create the environment using conda
```shell
conda env create -f IVP.yml
```

## Training and Inference
1. To train the IVP model on IISc VEED dataset,
```shell
cd src
python Trainer.py --configs configs/Configs_VEED.json --generate_data --correct_depth
cd ..
```

Our model generates some data before starting training. The `--generate_data` flag instructs the `Trainer.py` to generate this data. If the data has already been generated, this flag can be omitted.
Many datasets have errors in the depth maps, usually at the foreground-background boundaries. The `correct_depth` flag corrects this depth before further processing. If you have clean depth and do not want to employ depth correction, omit this flag. More details about depth correction can be found [here](https://github.com/NagabhushanSN95/IVP/src/data_generators/DepthCorrector.py).
To train on SceneNet dataset, use the configs [here](https://github.com/NagabhushanSN95/IVP/src/configs/Configs_Scenenet.json).

2. To run inference on IISc VEED dataset,
```shell
cd src
python Test.py --configs configs/Configs_VEED.json --generate_data --correct_depth --test_num 3
cd ..
```
The `configs` parameter and `generate_data` and `correct_depth` flags are similar to the training code.
The predicted frames will be saved in a folder named `Test0003`. You can specify different `test_num` if testing on different datasets with the same trained model.

## Citation

```
@inproceedings{kanchana2022ivp,
    title = {Revealing Disocclusions in Temporal View Synthesis through Infilling Vector Prediction},
    author = {Kanchana, Vijayalakshmi and Somraj, Nagabhushan and Yadwad, Suraj and Soundararajan, Rajiv},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    pages = {3541--3550},
    month = {January},
    year = {2022},
    doi = {10.1109/WACV51458.2022.00315}
}
```

## Acknowledgments
The code for depth based warping is borrowed from [here](https://github.com/NagabhushanSN95/Pose-Warping). 

For any queries or bugs related to either the IVP code or the IISc VEED database, please raise an issue.
