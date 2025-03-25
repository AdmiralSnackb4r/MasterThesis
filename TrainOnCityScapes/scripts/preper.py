

from CustomCocoDataset import Preparator


split_value = (0.6, 0.2, 0.2)

print(f"Creating Train and Test Datasets with split value of {split_value}")
preparator = Preparator(None, "../CityScapes/coco_annotations.json")
draws_train, draws_valid, draws_test = preparator.split_train_val_test(split=split_value)
preparator.create_split_annotations(draws_train, "train_dataset.json")
preparator.create_split_annotations(draws_valid, "valid_dataset.json")
preparator.create_split_annotations(draws_test, "test_dataset.json")