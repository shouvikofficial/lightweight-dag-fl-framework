from preprocessing.dataset_loader import (
    prepare_client_data
)

train_gen, val_gen, encoder = (
    prepare_client_data(
        "dataset/partitions/client_1.csv",
        "dataset/raw/ISIC_2019_Training_Input"
    )
)

print("Dataset Loaded Successfully")
print("Classes:", encoder.classes_)