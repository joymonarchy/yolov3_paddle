import paddle
from paddle import paddle
from Darknet53 import *
from toGetLable import *
from getP import *
from paddle.nn import Conv2D
                                                         

# 挑选出跟真实框IoU大于阈值的预测框

NUM_ANCHORS = 3
NUM_CLASSES = 7
num_filters=NUM_ANCHORS * (NUM_CLASSES + 5)

backbone = DarkNet53_conv_body()
detection = YoloDetectionBlock(ch_in=1024, ch_out=512)
conv2d_pred = Conv2D(in_channels=1024, out_channels=num_filters,  kernel_size=1)

x = paddle.to_tensor(img)
C0, C1, C2 = backbone(x)
route, tip = detection(C0)
P0 = conv2d_pred(tip)
# anchors包含了预先设定好的锚框尺寸
anchors = [116, 90, 156, 198, 373, 326]

total_loss = paddle.vision.ops.yolo_loss(P0,gt_boxes, gt_labels,anchors,[0,1],class_num = 7,ignore_thresh=0.7,
                                   downsample_ratio=8,
                                   use_label_smooth=True,
                                   scale_x_y=1.)
total_loss_data = total_loss.numpy()
print(total_loss_data)