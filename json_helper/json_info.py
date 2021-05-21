from collections import OrderedDict


class Json_data:
    def __init__(self):
        pass

    def __get_image_details_from_name(self, image_name):
        image_details = {}
        image_name = image_name.split(".")[0]

        image_details["pedestrian_id"] = image_name[0:4]
        for i in range(len(image_name)):
            if image_name[i] == 'C':
                image_details["camera_no"] = image_name[i: i+2]
            elif image_name[i] == 'T':
                image_details["tracklet_no"] = image_name[i: i+5]
            elif image_name[i] == 'F':
                image_details["frame_no"] = image_name[i:2]

        return image_details


    def info(self, num_classes):

        dataset_info = OrderedDict()
        dataset_info["name"] = "Mars Dataset"
        dataset_info["description"] = "N/A"
        dataset_info["url"] = "http://zheng-lab.cecs.anu.edu.au/Project/project_mars.html"
        dataset_info["version"] = "1.0"
        dataset_info["year"] = 2021
        dataset_info["contributor"] = "Sifat Ahmed"
        dataset_info["data_created"] = "2021/02/05"
        dataset_info["num_classes"] = num_classes

        return dataset_info


    def image(self, image_name, id, image_path):

        images_info = OrderedDict()
        images_info["license"] = 0
        images_info["image_id"] = id
        images_info["file_name"] = image_name
        images_info["path"] = image_path
        images_info["width"] = "128"
        images_info["height"] = "256"
        images_info["coco_url"] = "N/A"
        images_info["flickr_url"] = "N/A"

        return images_info

    def annotations(self, image_name, id, class_name):

        info_from_image_name = self.__get_image_details_from_name(image_name)

        annotation_info = OrderedDict()
        annotation_info["segmentation"] = []
        annotation_info["image_id"] = id
        annotation_info["class_name"] = class_name
        annotation_info["pedestrian_id"] = info_from_image_name['pedestrian_id']
        annotation_info["camera_no"] = info_from_image_name['camera_no']
        annotation_info["tracklet_no"] = info_from_image_name['tracklet_no']
        annotation_info["frame_no"] = info_from_image_name['frame_no']
        annotation_info["is_crowd"] = 0
        annotation_info["bbox"] = []

        return annotation_info
#
#     def image_details(self):
#         return self.__get_image_details_from_name("0065C1T0002F0016.jpg")
#
#
# if __name__ == "__main__":
#     j = Json_data()
#     print(j.image_details())