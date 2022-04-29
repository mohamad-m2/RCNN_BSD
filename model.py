import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
import torch.nn as nn



# let's make the RPN generate 5 x 3 anchors per spatial
# location, with 5 different sizes and 3 different aspect
# ratios. We have a Tuple[Tuple[int]] because each feature
# map could potentially have different sizes and
# aspect ratios
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                  aspect_ratios=((0.5, 1.0, 2.0),))

# let's define what are the feature maps that we will
# use to perform the region of interest cropping, as well as
# the size of the crop after rescaling.
# if your backbone returns a Tensor, featmap_names is expected to
# be ['0']. More generally, the backbone should return an
# OrderedDict[Tensor], and in featmap_names you can choose which
# feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)

mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=14,
                                                    sampling_ratio=2)
# put the pieces together inside a MaskRCNN model

def get_model(number_classes,model='resnset50'):
  # load a pre-trained model for classification and return
  # only the features
  if(model=='resnset50'):
    backbone=nn.Sequential(*list(torchvision.models.resnet50(pretrained=True).children())[:-2])
  else:
    backbone=nn.Sequential(*list(torchvision.models.resnet101(pretrained=True).children())[:-2])
  
  backbone.out_channels = 2048
  model = MaskRCNN(backbone,
                num_classes=number_classes,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
                mask_roi_pool=mask_roi_pooler)
  return model

