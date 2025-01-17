import sys

sys.path.insert(0, '.')
import torch
from torch.autograd import Variable
from torchvision.models import resnet
import pytorch_to_caffe
from MobileNetV2 import MobileNetV2

if __name__ == '__main__':
    name = 'MobileNetV2'
    net= MobileNetV2()
    checkpoint = torch.load("/home/shining/Downloads/mobilenet_v2.pth.tar")

    net.load_state_dict(checkpoint)
    net.eval()
    input = torch.ones([1, 3, 224, 224])
    # input=torch.ones([1,3,224,224])
    pytorch_to_caffe.trans_net(net, input, name)
    proto_path = '{}.prototxt'.format(name)
    print("Save the prototxt into {}".format(proto_path))
    pytorch_to_caffe.save_prototxt(proto_path)

    weight_path = '{}.caffemodel'.format(name)
    print("Save the caffe weight into {}".format(weight_path))
    pytorch_to_caffe.save_caffemodel(weight_path)
