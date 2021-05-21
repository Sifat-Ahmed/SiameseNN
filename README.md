# This repository contains the source code of Human Re-identification task.

## 1. Folder Structure
    + dataReader
        ---- dataset_reader.py
    + json_helper
        ---- json_creator.py
        ---- json_info.py
        ---- json_parser.py
    + loss
        ---- loss_func.py
    + models
        ---- resnet.py
        ---- siamese.py
        ---- siamese2.py
        ---- Siamese_EfficientNet.py
    ---- config.py
    ---- evaluator.py
    ---- helper.py
    ---- json_creator_main.py
    ---- main.py

## 2. How to Run
#### 2.1. Activating the Virtual Environment
    2.1.1. Move to the project directory.
        - project directory: ~/tokyo_project/Human-Re-identification/human_reid
        - venv directory: ~/tokyo_project/Human-Re-identification/human_reid/venv_torch
    2.1.2. Move to project directory then activate virtual Environment.
        - Command: source venv_torch/bin/activate

#### 2.2. Dataset Creation
- Dataset Directory: /data/public_data/MARS/Unzipped/bbox_train
- <b> command </b>: python json_creator_main.py 
- <b> parameters </b>:
    - [--train-json] :'path to train dataset folder'
    - [--train-output] : 'output filename/directory without .json'
    - [--num-train-classes] : 'number of training classes to take'
    
    - [--val-json] : 'path to val dataset folder'
    - [--val-output] : 'output filename/directory without .json'
    - [--num-val-classes] : 'number of validation classes to take'
    
    - [--test-json] : 'path to train dataset folder'
    - [--test-output] : 'output filename/directory without .json'
    - [--num-test-classes] : 'number of test classes to take'
    
    <b>Example: </b> python json_creator_main.py --train-json 'bbox_train' --train-output 'training_dataset' --num-train-classes 30

    >> This will create training_dataset.json in the default directory.\
    >> Validation and test dataset creation is same as above. Can be created simultaneously.
    
    >> By default it will pick classes that have more than 500 images and less than 1000 images.\
    >> To change this behaviour move to <b> json_helper/json_creator.py </b> Line No. 50\
    >> and edit the if condition. Has to be done manually. (TODO)

#### 2.3 Model Training
Here is a list of available models for training:
1. <b> Siamese </b> [Uses only convolution layers, no fully connected layers]
2. <b> SiameseNetwork </b> [Uses convolution with a fully connected layer]
3. <b> SiameseEfficientNet </b> [Uses efficientNet-b0 as feature extractor followed by a fully connected layer]
4. <b> ResNet50 </b> [Uses resnet50 as a feature extracture followed by a fully connected layer]
5. <b> ResNet101 </b> [Uses resnet101 as a feature extracture followed by a fully connected layer]
6. <b> ResNet152 </b> [Uses resnet152 as a feature extracture followed by a fully connected layer]

> - Model class definitions are in models folder. For any change required Please refer to that.
> - All the models with fully connected layers has 5 output neurons. This provides the optimal value. Some other values like 1, 2, 5, 8, 16, 32 has been tried before.

> - By default main.py will start training all the models. To change this behavious, please refer to <b> main.py, line no. 146 </b>. No command line argument added for this.

#### 2.3.1 How to run
>>  #### **config.py** contains all the configuration and parameters.
>>    1. image width
>>    2. image height
>>    3. learning rate
>>    4. epochs
>>    5. criterion
>>    6. train batch size
>>    7. validation batch size
>>    8. test batch size
>>    9. number of workers
>>    10. transform function
>>    11. optimizer
>>    12. learning rate scheduler
            
- First activate the venv as described in 2.1.2 unless not activated

- __command__ python main.py
- <b> parameters </b>:
    - [--train-json] 'Directory of the training Json' (**Required**)
    - [--val-json] 'Directory of the validation Json'
    
- If no validation json is given, it will use the training dataset to create a validation dataset
- **Example**: > python main.py --train-json 'dataset_train.json'

> #### To change any parameter, please refer to config.py
> By default all the above mentioned models will be trained for 50 epochs using the parameters defined in config.py
> Models will be saved according to their names and the best validation loss
> Loss Curves will be saved according to the model names.

#### 2.4 Testing
- __command__ python test.py
- <b> parameters </b>:
    - [--test-json] 'Directory of the testing Json file' (**Required**)
    - [--model-name] 'Name of the Model' (**Required**)
- Output:
    1. test loss
    2. Classification report
    3. ROC Curves
