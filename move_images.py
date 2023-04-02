import os
import shutil
from tqdm import tqdm

# images = os.listdir("images/moderate") 
# less_images = images[:500]


# for image in tqdm(less_images,total = (len(less_images))):
#     new_path = 'images/csr/' + image
    # shutil.move('images/moderate/'+image, new_path)

# images = os.listdir("images/csc") 
# less_images = images[:2500]


# for image in tqdm(less_images,total = (len(less_images))):
#     new_path = 'images/csr/' + image
#     shutil.move('images/csc/'+image, new_path)



import glob

folder_name = "normal"
removing_files = os.listdir(f"images/{folder_name}")[:500]
for i in removing_files:
    os.remove(f"images/{folder_name}/"+i)
