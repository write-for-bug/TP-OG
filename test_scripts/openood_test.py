# necessary imports
import torch
import sys
sys.path.append("./OpenOOD")
from openood.evaluation_api import Evaluator
from openood.networks import ResNet18_32x32 # just a wrapper around the ResNet

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load the model
    net = ResNet18_32x32(num_classes=10).to(device)
    net.load_state_dict(
    torch.load('./weights/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt',map_location=device)
)
    net.eval()
    postprocessor_name = "react" #@param ["openmax", "msp", "temp_scaling", "odin", "mds", "mds_ensemble", "rmds", "gram", "ebo", "gradnorm", "react", "mls", "klm", "vim", "knn", "dice", "rankfeat", "ash", "she"] {allow-input: true}

    evaluator = Evaluator(
                          net,
                          id_name='cifar10',                     # the target ID dataset
                          data_root='./datasets',                    # change if necessary
                          config_root=None,                      # see notes above
                          preprocessor=None,                     # default preprocessing for the target ID dataset
                          postprocessor_name=postprocessor_name, # the postprocessor to use
                          postprocessor=None,                    # if you want to use your own postprocessor
                          batch_size=200,                        # for certain methods the results can be slightly affected by batch size
                          shuffle=False,
                          num_workers=2)                         # could use more num_workers outside colab






