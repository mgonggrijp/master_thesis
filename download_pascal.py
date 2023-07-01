import wget
import os
import tarfile
import torch
import os
import shutil


# download and extract
if not os.path.exists("datasets/pascal/data/VOCtrainval_11-May-2012.tar"):
    url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    output_dir = "datasets/pascal/data"
    os.makedirs(output_dir, exist_ok=True)
    print("Downloading PASCAL VOC2012 dataset...")
    wget.download(url, out=output_dir)
    print("\nDownload complete.")
    print("\nExtracting files...")
    tar_file_path = "datasets/pascal/data/VOCtrainval_11-May-2012.tar"
    tar = tarfile.open(tar_file_path)
    tar.extractall("datasets/pascal/data")
    tar.close()
    print("\nExtraction complete.")


# move subfolders to pascal/data
if not os.path.exists("datasets/pascal/data/Annotations"):
    print('\nMoving data sub folders to pascal/data..')
    source_folder = "datasets/pascal/data/VOCdevkit/VOC2012"  
    destination_folder = "datasets/pascal/data"  
    folders_to_move = os.listdir(source_folder)  
    for folder_name in folders_to_move:
        source_path = os.path.join(source_folder, folder_name)
        destination_path = os.path.join(destination_folder, folder_name)
        shutil.move(source_path, destination_path)
    shutil.rmtree("datasets/pascal/data/VOCdevkit")
    print('\nDone.')
    

# compute class weights from their inverse sample frequencies
if not os.path.exists("datasets/pascal/class_weights.pt"):
    print("\nComputing class weights..")
    file_path = "datasets/pascal/pascal_class_distributions.txt"  
    # compute the inverse weights and stores them as a tensor
    num_classes = 21
    class_weights = torch.zeros(num_classes)
    with open(file_path, "r") as file:
        for i, line in enumerate(file):
            line = line.strip()  # Remove leading/trailing whitespaces
            split_items = line.split()  # Split the line by whitespace
            if len(split_items) >= 6:  # Ensure the line has at least 6 items
                class_count = int(split_items[5])  # Select the sixth item (0-based index)
                class_weights[i + 1] = (1 / class_count) * 100
    # unlabeled / background is in every image;
    class_weights[0] = (1/class_weights.sum()) 
    print(class_weights)
    # Specify the file path
    file_path = "datasets/pascal/class_weights.pt"
    # Save the tensor
    torch.save(class_weights, file_path)
    print('\nDone.')

    