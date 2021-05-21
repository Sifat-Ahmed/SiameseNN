import argparse
import helper
from config import Config
from json_helper.json_parser import ParseJson
from dataReader.dataset_reader import CreateDataset
from torch.utils.data import DataLoader
from models.resnet import ResNet50, ResNet101, ResNet152
from models.siamese2 import SiameseNetwork
from models.Siamese_EfficientNet import SiameseEff
from loss.loss_func import ContrastiveLoss
from models.siamese import Siamese
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np

#criterion = torch.nn.CosineEmbeddingLoss()
DEVICE = helper.get_device()
SAVE_MODEL = np.inf

def train(model, train_loader, val_loader,  cfg, training_loss_graph, val_loss_graph, name):
    global SAVE_MODEL
    SAVE_MODEL = np.inf
    optimizer = cfg.set_get_optim(model.parameters(),
                                       cfg.learning_rate)
    lr_scheduler = cfg.get_lr_scheduler(optimizer)

    for i in range(0, cfg.epochs):
        train_loss = 0.0
        epoch_loss = 0.0
        predictions = list()
        true_labels = list()
        model.train()

        for img1, img2, lbl in tqdm(train_loader):
            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)
            lbl = lbl.to(DEVICE)
            optimizer.zero_grad()
            output1, output2, output = model(img1, img2)
            if cfg.criterion.__class__.__name__ == "ContrastiveLoss":
                loss = cfg.criterion(output1, output2, lbl)
                loss.backward()
                train_loss += loss.item()
            elif cfg.criterion.__class__.__name__ == "HingeEmbeddingLoss":
                loss = cfg.criterion(output, lbl)
                loss.backward()
                train_loss += loss.item()


            optimizer.step()

            #pred = F.pairwise_distance(output1, output2)

            lr_scheduler.step()
        epoch_loss = train_loss / len(train_loader)
        #print("Training Accuracy", accuracy_score(y_true=true_labels, y_pred=predictions))
        training_loss_graph.append(epoch_loss)
        print('Training Epoch: {}/{} || Loss: {:.4f} '.format(i+1, cfg.epochs, epoch_loss))

        validation(model, val_loader, cfg, val_loss_graph, name)


def validation(model, val_loader, cfg, val_loss_graph, name):
    model.eval()
    val_loss, epoch_vloss = 0.0, 0.0
    predictions = list()
    true_labels = list()

    for img1, img2, lbl in tqdm(val_loader):
        img1, img2, lbl = img1.to(DEVICE), img2.to(DEVICE), lbl.to(DEVICE)

        output1, output2, output = model(img1, img2)

        if cfg.criterion.__class__.__name__ == "ContrastiveLoss":
            loss = cfg.criterion(output1, output2, lbl)
            loss.backward()
            val_loss += loss.item()
        elif cfg.criterion.__class__.__name__ == "HingeEmbeddingLoss":
            loss = cfg.criterion(output, lbl)
            loss.backward()
            val_loss += loss.item()



    epoch_vloss = val_loss / len(val_loader)
    val_loss_graph.append(epoch_vloss)
    #print("Validation Accuracy", accuracy_score(y_true=true_labels, y_pred=predictions))
    print('Validation || Loss: {:.4f} '.format(epoch_vloss))

    #print(predictions)

    global SAVE_MODEL
    if epoch_vloss < SAVE_MODEL:
        SAVE_MODEL = epoch_vloss
        torch.save(model.state_dict(), name+'.pth')
        print('Model Saved!')

def main():
    args = helper.parse_args()
    cfg = Config(args)




    print("Parsing JSON...")
    json_dataset_parser = ParseJson()

    print("Parsing Finished.. \nCreating Train/Val sets")
    train_img, train_lbl, val_img, val_lbl = json_dataset_parser.get_train_val(args.train_json, args.val_json)


    #print(np.unique(train_lbl, return_counts=True))

    train_dataset = CreateDataset(train_img,
                                  train_lbl,
                                  transform=cfg.transform)
    val_dataset = CreateDataset(val_img,
                                val_lbl,
                                transform=cfg.transform)


    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train_batch,
                              num_workers=cfg.num_workers,
                              shuffle=True,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.val_batch,
                            num_workers=cfg.num_workers,
                            shuffle=True,
                            pin_memory=True)

    # for img1, img2, lbl in train_loader:
    #     print(lbl)

    #
    #
    # #model = Cnn()

    models = [Siamese(), SiameseNetwork(), SiameseEff(), ResNet50(), ResNet101(), ResNet152()]
    names =  ["Siamese", "SiameseNetwork", "SiameseEfficientNet", "ResNet50", "ResNet101", "ResNet152"]

    for model, name in zip(models, names):
        training_loss_graph = list()
        val_loss_graph = list()

        #model = SiameseNetwork()
        model = model.to(DEVICE)

        if os.path.isfile(name+'.pth'):
            print('Saved Model found. Loading...')
            model.load_state_dict(torch.load(name+'.pth'))

        print("Training started")
        train(model,
               train_loader,
               val_loader,
               cfg, training_loss_graph, val_loss_graph, name)

        print('Finished Training...')

        epochs = range(1, cfg.epochs+1)

        plt.figure(figsize=(7,7))
        plt.plot(epochs, training_loss_graph, 'g', label='Training loss')
        plt.plot(epochs, val_loss_graph, 'b', label='validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.rcParams["font.size"] = "20"
        plt.savefig('loss curve '+ name +'.png')
    #plt.show()


if __name__ == '__main__':
    main()
