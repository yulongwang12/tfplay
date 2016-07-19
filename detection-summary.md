There are two established classes of methods for object detection in images
* sliding windows
  * Deformable Part Model (DPM)
  * OverFeat
  * YOLO
  * SSD
  * G-CNN

* region proposal classification
  * MultiBox
  * R-CNN
  * SPPnet
  * MR-CNN
  * Fast R-CNN
  * ION
  * Faster R-CNN

* bounding box
  * Selective Search
  * LocNet

###MR-CNN
1. Multi-region + crop pooling to fixed size + concatenate
2. semantic segmentation-aware: use FCN structure to predict fg & bg
