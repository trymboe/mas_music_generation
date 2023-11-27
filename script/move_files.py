import os
import shutil

source_dir = "data/POP909"
destination_dir = "data/POP909/transposed"

for folder in os.listdir(source_dir):
    if folder == ".DS_Store":
        continue
    source_path = os.path.join(source_dir, folder, "beat_audio.txt")
    destination_path = os.path.join(destination_dir, folder)

    if os.path.exists(destination_path):
        print("Copying files from", source_path, "to", destination_path)
        shutil.move(source_path, destination_path)
