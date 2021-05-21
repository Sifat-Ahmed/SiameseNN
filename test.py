import argparse
import helper
from config import Config
from json_helper.json_parser import ParseJson
from dataReader.dataset_reader import CreateDataset
from torch.utils.data import DataLoader
from models.siamese import Siamese
from models.siamese2 import SiameseNetwork
from models.Siamese_EfficientNet import SiameseEff
from models.resnet import ResNet50, ResNet101, ResNet152
from loss.loss_func import ContrastiveLoss
import torch.nn.functional as F
import numpy as np
from evaluator import get_roc_curve
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import os

DEVICE = helper.get_device()

def test(model, test_loader, cfg, test_data, model_name):
    model.eval()
    original_lbl, pred_lbl = list(), list()
    test_loss = 0.0

    for img1, img2, lbl in tqdm(test_loader):
        img1, img2, lbl = img1.to(DEVICE), img2.to(DEVICE), lbl.to(DEVICE)
        output1, output2, output = model(img1, img2)

        if cfg.criterion.__class__.__name__ == "ContrastiveLoss":
            loss = cfg.criterion(output1, output2, lbl)
            loss.backward()
            test_loss += loss.item()
            output = F.pairwise_distance(output2, output1)
        elif cfg.criterion.__class__.__name__ == "HingeEmbeddingLoss":
            loss = cfg.criterion(output, lbl)
            loss.backward()
            test_loss += loss.item()

        for out, lb in zip(output, lbl):
            pred = out.item()
            pred_lbl.append(pred)
            original_lbl.append(int(lb))


    _tloss = test_loss / len(test_loader)

    print('Test Loss:', _tloss)

    #for i in range(len(original_lbl)):
        #print(test_data[i], pred_lbl[i])

    #print(original_lbl, pred_lbl)

    get_roc_curve(original_lbl,
                  pred_lbl,
                  os.path.join(cfg.save_roc_path, model_name))

def main():
    args = helper.parse_test()
    cfg = Config(args)

    print("Parsing JSON...")
    json_dataset_parser = ParseJson()

    print("Parsing Finished.. \n")

    test_img, test_lbl = json_dataset_parser.get_test(args.test_json)
    # print(len(train_img), len(val_img), len(test_img))

    # train_dataset = CreateDataset(train_img,
    #                               train_lbl,
    #                               transform=cfg.transform)
    # val_dataset = CreateDataset(val_img,
    #                             val_lbl,
    #                             transform=cfg.transform)

    test_dataset = CreateDataset(test_img,
                                 test_lbl,
                                 transform=cfg.transform)

    # train_loader = DataLoader(train_dataset,
    #                           batch_size=cfg.train_batch,
    #                           num_workers=cfg.num_workers,
    #                           shuffle=True)
    #
    # val_loader = DataLoader(val_dataset,
    #                         batch_size=cfg.val_batch,
    #                         num_workers=cfg.num_workers)

    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.test_batch)

    # for img, img2, lbl in train_loader:
    #
    #     #helper.visualize_tensor_image(img[0])
    #     #break
    #     pass

    # print(num_classes)

    models = {
        'Siamese': Siamese(),
        'SiameseNetwork': SiameseNetwork(),
        'SiameseEfficientNet': SiameseEff(),
        'ResNet50': ResNet50(),
        'ResNet101': ResNet101(),
        'ResNet152': ResNet152()

    }

    model = models[args.model_name]
    model = model.to(DEVICE)
    #model = vgg.get_model(num_classes).to(DEVICE)

    # resnet = Resnet50(pretrained=False)
    # model = resnet.get_model(num_classes).to(DEVICE)
    #
    print(args.model_name+'.pth')

    model_path = os.path.join(cfg.save_model_path, args.model_name+'.pth')

    if os.path.isfile(model_path):
        print('Saved Model found. Loading...')
        model.load_state_dict(torch.load(model_path))
    else:
        raise("Model not found, names are '['Siamese', 'SiameseNetwork', 'SiameseEfficientNet', 'ResNet50', 'ResNet101', 'ResNet152']")
    #
    #


    print('Testing...')

    test(model,
            test_loader,
            cfg,
            test_img,
            model_name=args.model_name)

if __name__ == '__main__':
    main()
