from CustomCocoDataset import Preparator


print(f"Creating Train and Test Datasets with split value of {500/500}")
preparator = Preparator(None, "CityScapes\\annotations\\coco_annotations.json", exclude_category_ids=(
    4, 13, 14, 15, 5, 6, 22, 9, 21, 3, 7))
draws_train, draws_valid, draws_test = preparator.split_train_val_test()
preparator.create_split_annotations(draws_train, "train_dataset.json")
preparator.create_split_annotations(draws_valid, "valid_dataset.json")
preparator.create_split_annotations(draws_test, "test_dataset.json")