import os
import json
from json_helper.json_info import Json_data
from tqdm import tqdm
import random
from collections import OrderedDict

class JsonCreator:
    def __init__(self,
                 data_path,
                 output_filename,
                 ):
        self._data_path = data_path
        self._output_filename = output_filename
        self._classes = os.listdir(self._data_path)
        self._json_info = Json_data()
        self._dataset_list = list()


    def __get_path__(self, path_1, path_2):
        return os.path.join(path_1, path_2)

    def __create_dataset__(self):
        # self._info = self._json_info.info(len(self._classes))
        # self._images_list, self._annotations_list = list(), list()
        # count = 0
        #
        # for class_ in tqdm(self._classes):
        #     class_path = self.__get_path__(self._data_path, class_)
        #
        #     images = os.listdir(class_path)
        #     for img in images:
        #         image_path = os.path.join(class_path, img)
        #         self._images_list.append(self._json_info.image(img, count, image_path))
        #         self._annotations_list.append(self._json_info.annotations(img, count, class_))
        #         count += 1
        #
        # self._dataset ={
        #     "info": self._info,
        #     "images": self._images_list,
        #     "annotations": self._annotations_list
        # }
        size = 0

        for cls_ in tqdm(self._classes[30:35]):

            images = os.listdir(os.path.join(self._data_path, cls_))
            x = [i for i in range(0, len(images)-1, 5)]
            y = [i for i in range(len(images)-1, 0, -5)]
            #y = [i for i in range(0, len(images) - 1, 10)]

            for i, j in zip(x, y):
                #print(i, j)

                image_dict = OrderedDict()
                image_dict["image_1"] = os.path.join(self._data_path, cls_, images[i])
                image_dict["image_2"] = os.path.join(self._data_path, cls_, images[j])
                image_dict["class"] = 0.0

                # print(image_dict)
                self._dataset_list.append(image_dict)
                size += 1
        print("Similar images", size)

        count = 0

        while True:
            class_1 = random.choice(self._classes[30:35])
            class_2 = random.choice(self._classes[30:35])

            if class_1 != class_2:
                images_1 = os.listdir(os.path.join(self._data_path, class_1))
                images_2 = os.listdir(os.path.join(self._data_path, class_2))

                image_dict = OrderedDict()
                image_dict["image_1"] = os.path.join(self._data_path,class_1, random.choice(images_1))
                image_dict["image_2"] = os.path.join(self._data_path, class_2, random.choice(images_2))
                image_dict["class"] = 1.0

                # print(image_dict)
                self._dataset_list.append(image_dict)
                count += 1

                if count == size:
                    print("Total images:", count+size)
                    break

        self._dataset = {
            'dataset': self._dataset_list
        }

        self.__write_on_dir(self._dataset)


    def __write_on_dir(self, json_file):
        fw = open(self._output_filename+'.json', 'w')
        json.dump(json_file, fw, indent=2)


    def make(self):
        self.__create_dataset__()


if __name__ == "__main__":

    print(os.getcwd())
    j = JsonCreator(r'H:\Research\JTEKT\Human reid\Dataset\bbox_train', 'dataset_val')
    j.make()