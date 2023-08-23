import os
# from tqdm import tqdm
root_dir = "images"
output_file = "image_paths.txt"
for dir in os.listdir(root_dir):
    if os.path.isdir(os.path.join(root_dir, dir)):
        for file in os.listdir(os.path.join(root_dir, dir)):
            with open(output_file, 'a') as f:
                f.write(os.path.join(dir, file) + '\n')