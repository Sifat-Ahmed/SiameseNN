import gc
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument('--train-json',
                        help='Json File for training',
                        required=True,
                        type=str)

    parser.add_argument('--val-json',
                        help='Json File for validation',
                        required=False,
                        type=None)

    parser.add_argument('--test-json',
                        help='Json file for testing',
                        type=str,
                        default=None)

    parser.add_argument('--epochs',
                        help='log directory',
                        type=int,
                        default=100)

    parser.add_argument('--model-name',
                        help='vgg19 / resnet50',
                        required=False,
                        type=str)

    return parser.parse_args()


def parse_test():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument('--test-json',
                        help='Json File for testing',
                        required=True,
                        type=str)

    parser.add_argument('--model-name',
                        help='',
                        required=True,
                        type=str)

    return parser.parse_args()

def check_create_paths(model_path, loss_path, roc_path):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(loss_path):
        os.makedirs(loss_path)
    if not os.path.exists(roc_path):
        os.makedirs(roc_path)


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def empty_cache():
    torch.cuda.empty_cache()


def call_gc():
    gc.collect()


def set_parameter_requires_grad(model, requires_grad):
        for param in model.parameters():
            param.requires_grad = requires_grad

def visualize_tensor_image(image):

    image = transforms.ToPILImage()(image).convert('RGB')
    plt.imshow(image)
    plt.show()
