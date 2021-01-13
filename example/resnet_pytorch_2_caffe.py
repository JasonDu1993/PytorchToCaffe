import sys

sys.path.insert(0, '.')
import torch
from torch.autograd import Variable
from torchvision.models import resnet
import pytorch_to_caffe

if __name__ == '__main__':
    name = 'resnet18'
    resnet18 = resnet.resnet18(pretrained=True)
    # checkpoint = torch.load("/root/.cache/torch/checkpoints/resnet18-5c106cde.pth")

    # resnet18.load_state_dict(checkpoint)
    resnet18.eval()
    input = torch.ones([1, 3, 224, 224])
    pytorch_to_caffe.trans_net(resnet18, input, name)
    proto_path = '{}.prototxt'.format(name)
    print("Save the prototxt into {}".format(proto_path))
    pytorch_to_caffe.save_prototxt(proto_path)

    weight_path = '{}.caffemodel'.format(name)
    print("Save the caffe weight into {}".format(weight_path))
    pytorch_to_caffe.save_caffemodel(weight_path)
