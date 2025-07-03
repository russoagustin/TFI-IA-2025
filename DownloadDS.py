import kagglehub
# Download latest version
base_dir = kagglehub.dataset_download("ashery/chexpert")
base_dir = base_dir + '/'

print("Path to dataset files:", base_dir)