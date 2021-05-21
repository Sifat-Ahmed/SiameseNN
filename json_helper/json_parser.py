import json
from sklearn.model_selection import train_test_split
import numpy as np

class ParseJson:

    def __init__(self):
        pass

    def _parse_json(self, json_file):
        images, labels = list(), list()

        with open(json_file, "r") as f:
            data = json.load(f)

            #num_classes = data['info']['num_classes']

            for data in data['dataset']:
                #print(data['image_1'])
                #print()

                images.append([data['image_1'], data['image_2']])
                labels.append(data['class'])

        #print(images)

        return images, labels #, num_classes

    def get_train_val(self, train_dataset, val_dataset=None, val_size = 0.1):
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset

        if self._train_dataset:
            self._x_train, self._y_train = self._parse_json(self._train_dataset)


        if self._val_dataset:
            self._val_img, self._val_lbl = self._parse_json(self._val_dataset)
        else:
            self._x_train, self._val_img, self._y_train, self._val_lbl = train_test_split(self._x_train,
                                                                                              self._y_train,
                                                                                              test_size=val_size,
                                                                                              shuffle=True,
                                                                                              stratify=self._y_train)


        return self._x_train, self._y_train, self._val_img, self._val_lbl

    def get_test(self, test_dataset=None):
        self._test_dataset = test_dataset

        self._test_img, self._test_lbl = self._parse_json(self._test_dataset)
        return self._test_img, self._test_lbl




if __name__ == "__main__":
    parser = ParseJson()
    x1, y1, v1, v2 = parser.get_train_val("")

    for i in range(len(x1)):
        print(x1[i])
        print(y1[i])