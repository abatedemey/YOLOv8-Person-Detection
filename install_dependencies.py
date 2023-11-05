import subprocess
import os

# Define the URL of the file to download
file_url_sam = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
file_url_yolov8 = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
)

# Define the directory where you want to save the file
destination_directory = "./models"

# Ensure the directory exists
os.makedirs(destination_directory, exist_ok=True)

# Download the file using wget
subprocess.run(["wget", file_url_sam, "-P", destination_directory])
subprocess.run(["wget", file_url_yolov8, "-P", destination_directory])
