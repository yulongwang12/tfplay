# Data
put a symbolic link under `datasets` with name `VOCdevkit2007`

Below consider `experiments/script_faster_rcnn_VOC2007_ZF` as example

# Dataset Preparation
* `Dataset.voc2007_trainval` under `experiment/+Dataset` 
  * call `imdb_from_voc` from `imdb/`, collect imdb information and image list, decide whether to flip the image
  * call the imdb's `roidb_func` to get roi infos
  * imdb cache in `imdb/cache`

* `proposal_config` under `functions/rpn`, configure proposal parameters, including
  * `drop_boxes_runoff_image`: discard boxes outside image
  * `scales`/`max_size`: 600/1000, fixed the shorter side and longer side
  * `ims_per_batch = 1`: only support 1 image per batch
  * `batch_size`: final extracted region numbers
  * `fg_fraction`/`bg_weight`/`bg_thres_hi`/`bg_thres_lo`: for deciding to choose proposals
  * `test_scales`/`test_max_size`: testing image size
  * `test_nms`: 0.3 for testing nms
  * `test_min_box_size`: 16
  * `test_drop_boxes_runoff_image`: false

* `faster_rcnn_config` under `functions/faster_rcnn`
  * `ims_per_batch`: 2, perhaps proposal network only consider 1 image input, and faster rcnn feed 2 image per iteration

* `Faster_RCNN_Train.set_cache_folder` under `experiment/+Faster_RCNN_Train`, set cache for 4 stages when training  Faster RCNN

* `proposal_prepare_anchors` output anchors to `conf_proposal` variable
  * first calculate output size `proposal_calc_output_size` under `functions/rpn`
    * seems to iterate through input size from 100 to `conf.max_size` and get the output width and height from `caffe_net.blobs('proposal_cls_score')`, and store the input key and output value in `output_width_map` and `output_height_map`
  * `proposal_generate_anchors` under `functions/rpn` with `scales=[8, 16, 32]`
    * output precomputed anchors to `anchor_cache_dir` in `output/rpn_cachedir/some-name`
    * first `ratio_jitter`, then `scale_jitter`

# Stage 1 proposal
* `Faster_RCNN_Train.do_proposal_train`, args: `conf_proposal`, `dataset`, `model.stage1_rpn`, `opt.do_val=True`, returns: `model.stage1_rpn`
  * call `proposal_train` under `functions/rpn`
  * `solver_def_file` and `net_file` in models dir
  * prepare training data with `proposal_prepare_image_roidb`, return `image_roidb_train`, `bbox_means`, `bbox_stds`
  * `generate_random_minibatch` to variable `shuffled_inds` and `sub_db_inds`
  * `proposal_generate_minibatch` from `image_roidb_train(sub_db_inds)` to variable `net_inputs` and `scale_inds`
  * Then feed input to net, train for one iteration 

* `Faster_RCNN_Train.do_proposal_test`: use `model.stage1_rpn` to extract `dataset.roidb_train` for next stage training

# Stage 1 fast rcnn training
* `Faster_RCNN_Train.do_fast_rcnn_train`, args: `conf_fast_rcnn`, `dataset`, `model.stage1_fast_rcnn`, `opt_do_val=True`. Now `dataset.roidb_train` has gathered rpn results

**Notice**: Meanwhile, the model has performed test `Faster_RCNN_Train.do_fast_rcnn_test`, get results in `opts.mAP`

# Stage 2 proposal
* init `model.stage2_rpn` with `model.stage1_fast_rcnn.output_model_file`
* only finetune the final rpn net
* repeat Stage 1 proposal

# Stage 2 fast rcnn training
* init `model.stage2_fast_rcnn` with `model.stage1_fast_rcnn.output_model_file`
* repeat Stage 1 training

# Perform Final test
* on `dataset.roidb_test`

# Gather models
* `Faster_RCNN_Train.gather_rpn_fast_rcnn_models`
