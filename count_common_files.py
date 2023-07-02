import os

def get_file_names(folder_path):
    file_names = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_name = os.path.splitext(file)[0]
            file_names.append(file_name)
    return file_names

def count_common_files(folder1_path, folder2_path):
    folder1_files = get_file_names(folder1_path)
    folder2_files = get_file_names(folder2_path)
    
    common_files = set(folder1_files) & set(folder2_files)
    return len(common_files)

# Specify the paths to the folders
folder1_path = 'datasets/pascal/data/SegmentationClassAug'
folder2_path = 'datasets/pascal/data/JPEGImages'

# Count the common files
common_files_count = count_common_files(folder1_path, folder2_path)
print(f"Number of common files: {common_files_count}")