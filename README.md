# Supervised Contrastive Learning & Knowledge Distillation version for LTML

This is the code for improving the model of LTML with Supervised Contrastive Learning and Knowledge Distillation.
 
The Model is improved from the work "Long-Tailed Multi-Label Visual Recognition by Collaborative Training on Uniform and Re-balanced Samplings" by Hao Guo

## Requirements 
* [Pytorch](https://pytorch.org/)
* [Sklearn](https://scikit-learn.org/stable/)




### Use our dataset
The long-tail multi-label datasets we use in the paper are created from [MS COCO](https://cocodataset.org/) 2017 and [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) 2012. Annotations and statistics data resuired when training are saved under `./appendix` in this repo.
```
appendix
  |--coco
    |--longtail2017
      |--class_freq.pkl
      |--class_split.pkl
      |--img_id.pkl
  |--VOCdevkit
    |--longtail2012
      |--class_freq.pkl
      |--class_split.pkl
      |--img_id.pkl
```
