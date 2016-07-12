# Overall Structure
* caffe-faster-rcnn: modified caffe, including special layers (ROIPoolingLayer)
* lib: Cython modules
* data: store pascal data & trained models
  * `ln -s VOCdevkit /path/to/downloaded/VOCdevkit2007`

# Training procedure
```shell
./experiments/scripts/faster_rcnn_end2end.sh [GPU_ID] [NET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
# --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

* designate log file to `experiments/logs/faster_rcnn_end2end_---.txt`
* 
```shell
time ./tools/train_net.py --gpu --solver --weights --imdb --iters --cfg
```
  * `solver`: point to `models/${DATASET}/${NET}/faster_rcnn_end2end/solver.prototxt`, 
        the corresponding net.prototxt is also included.
  * `weights`: pretrained models, downloaded to `data/imagenet_models/${NET}.caffemodel`
  * `imdb`: train imdb
  * `cfg`: point to `experiments/cfgs/faster_rcnn_end2end.yml`

# Construct imdb
In `tools/train_net.py`
* `train_net`: imported from `lib/fast_rcnn/train`
  * only receive precomputed roi data for roi pooling
  * before feedint to solver, need to filter_roidb
    - filter out `max_overlaps` bigger than `cfg.TRAIN.FG_THRESH` 
    or between `cfg.TRAIN.BG_THRESH_HI` and `cfg.TRAIN.BG_THRESH_LO` 
  * roidb for `SolverWrapper` construction, but here are some *configurations*:
    - 
    ```python
    if (cfg.TRAIN.HAS_RPN and cfg.TRAIN.BBOX_REG and
        cfg.TRAIN.BBOX_NORMALIZE_TARGETS):
        assert cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED

    if cfg.TRAIN.BBOX_REG:
        print 'Computing bounding-box regression targets...'
        self.bbox_means, self.bbox_stds = \
                rdl_roidb.add_bbox_regression_targets(roidb)
        print 'done'
    ```
    - `rdl_roidb` imported from `lib/roi_data_layer/roidb`

* before training net, `imdb, roidb = combined_roidb(imdb_name)`
* `get_training_roidb`: imported from `lib/fast_rcnn`
* `get_imdb`: imported from `lib/datasets/factory`
* `imdb`: imported from `lib/dataset/imdb`
