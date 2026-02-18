import os
# Import the COLMAP utility script (assuming it's in the same directory)
import dataset.read_write_model as rw

# Define the path to your directory containing the cameras.bin file
input_model_dir = '/home/anurag/stream_captures/sparse/0'
cameras_bin_path = os.path.join(input_model_dir, 'cameras.bin')
extrinscs_bin_path = os.path.join(input_model_dir, 'images.bin')
# Read the cameras data from the binary file
cameras = rw.read_cameras_binary(cameras_bin_path)
extrinsics = rw.read_images_binary(extrinscs_bin_path)
# Now you can access the camera data
print(f"Loaded {len(cameras)} cameras.")
print(cameras)
print(extrinsics)

