import pixellib
from pixellib.custom_train import instance_custom_training

train_maskrcnn = instance_custom_training()
train_maskrcnn.modelConfig(network_backbone="resnet101", num_classes=4, batch_size=15)
train_maskrcnn.load_pretrained_model("mask_rcnn_coco.h5")
train_maskrcnn.load_dataset("images_segmentation_3")
train_maskrcnn.train_model(num_epochs=3, augmentation=True, path_trained_models="junctions_last_c")