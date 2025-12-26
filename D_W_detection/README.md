### Yolov8 doors, windows detection

# splitter
    
    split_1024.py --input_dir <input> --output_dir <output_1024> --overlap <overlap size default 0> 

# dataset_config

    in file data.yaml

# available datasets

    small_dataset_1024 : the model is trained on this dataset.
    All_plans_Full : All available plans 
    All_plans_1024 : All plains splitted in 1024x1024 format 
    dataset_Full_plans : images + labels small dataset of Full_Plans (not needed in this program)
    small_test_dataset_1024 : few images in 1024 format 

# prediction results
    
    in runs\detect:
        predict : prediction of small_dataset_1024 (training dataset)
        predict2: prediction of small_test_dataset_1024
        predict3: prediction of All_plans_1024 
    
# Training 
        model is trained on small_dataset_1024.
        in file data.yaml change training and validation dataset.
        run train.py

# Prediction 
        with predict.py

# notice 
        All_plan_1024 is under annotation      