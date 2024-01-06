# Brain Tumor Segmentation Projects
Repository to predict brain tumor using otsu image segmentation and advance computer vision techniques to get the prediction from MRI Brain Images.

## Repository Structure
This repository consists python script and notebook to run brain tumor segmentation, example input directory, and example output from this program. Here is the repository structure:
```bash
.
├── brain_tumor_segmentation.ipynb # Jupyter notebook
├── brain_tumor_segmentation.py # Python Script
├── dataset # Example input
├── README.md # Readme
├── requirements.txt # Requirements
├── result_prediction # Example output directory
└── result_prediction.csv # Example output csv files
```

## How to Use?

### Install Requirements

#### Miniconda Virtual Environment
First, you need to install miniconda dependecies if you have'nt install python to create python virtual environment.
- [https://docs.conda.io/projects/miniconda/en/latest/](miniconda docs)

You can simply run this command to create new environment.
```bash
conda create -n <environment-name>

# example
conda create -n brain-tumor-environment python==3.12
```

Then you can activate the environment like this.
```bash
conda activate brain-tumor-environment
```

#### Install Python Package

After you enter the python virtual environment, you can run this command to install all dependencies.
```
pip3 install -r requirements.txt
```

### Run the Program
The python scripts have 2 arguments like this following section.
```bash
usage: brain_tumor_segmentation.py [-h] -d DIRECTORY [-t TARGET_DIRECTORY]

Brain Tumor Segmentation

options:
  -h, --help            show this help message and exit
  -d DIRECTORY, --directory DIRECTORY
                        path to input directory images
  -t TARGET_DIRECTORY, --target_directory TARGET_DIRECTORY
```

You can run the script like this.
```bash
python3 brain_tumor_segmentation -d INPUT_DIRECTORY [-t TARGET_DIRECTORY] # target_directory is optional.
```

Example:
```bash
 python3 brain_tumor_segmentation.py -d dataset/
```

Output:
Output for this program is one directory of prediction result and csv file.

```bash
├── result_prediction # Example output directory
│   ├── muryadi 10-img-00001-00001 (1)_extracted_tumor_img.bmp # output extracted tumor
│   ├── muryadi 10-img-00001-00001 (1)_pred_result.bmp # output visualization that consist bbox and segmented area in original image.
└── result_prediction.csv # Example output csv files
```