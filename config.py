import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from loss.loss_func import ContrastiveLoss

class Config:
    def __init__(self, args):
        self.image_width = 64
        self.image_height = 64
        self.learning_rate = 0.005
        self.epochs = 50
        self.criterion = nn.HingeEmbeddingLoss()#nn.CosineEmbeddingLoss() #nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss()
        #self.criterion = ContrastiveLoss()
        self.train_batch = 16
        self.val_batch = 8
        self.test_batch = 8
        self.num_workers = 4

        self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.image_width, self.image_height)),
                transforms.ToTensor(),
                #transforms.Normalize([0.5, 0.5, 0.5])
            ])



    def set_get_image_height_width(self, height, width):
        self.image_width = width
        self.image_height = height

        return self.image_width, self.image_height

    def set_get_optim(self, model_params, lr_rate = None, betas = (0.9, 0.999) ):
        if lr_rate is None:
            lr_rate = self.learning_rate
        else:
            self.learning_rate = lr_rate

        self.optimizer = optim.Adam(model_params, lr=lr_rate, weight_decay=0.0005)

        return self.optimizer

    def get_lr_scheduler(self, optimizer, step_size = 7, gamma = 0.1):
        self.exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return self.exp_lr_scheduler