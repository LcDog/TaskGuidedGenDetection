# PyTorch Faster R-CNN code for running on SIM10K and CityScapes_Car

- Run `train_ori.py` to train on SIM10K and validate on cs_car for every `--eval_interval` training steps.
- Run `train_ori_no_eval.py` to just train on SIM10K.
- Run `test_net_list.py` to evaluate the models saved in some specific folder.

## Install

The code is based on [PyTorch-FasterRCNN](https://github.com/jwyang/faster-rcnn.pytorch). Please follow their notes to install (compile) it.