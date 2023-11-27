import os

for idx, directory in enumerate(os.listdir("data/POP909/transposed")):
    if ".DS_Store" in directory:
        continue
    new_name = directory.split(".")[0]
    root_dir = "data/POP909/transposed"

    os.rename(os.path.join(root_dir, directory), os.path.join(root_dir, new_name))
