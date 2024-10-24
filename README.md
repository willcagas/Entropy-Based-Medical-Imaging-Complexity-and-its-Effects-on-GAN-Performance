# Synthetic Medical Image Generation

This is the supplementary material for the [*Medical Imaging Complexity and its Effects on GAN Performance*](https://arxiv.org/abs/arXiv:2410.17959) paper. The code is based on different approaches, including **SPADE-GAN**, **StyleGAN 3**, and utilizes **TorchXRayVision**.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Results](#results)
- [References](#references)
- [Acknowledgements](#acknowledgments)
- [Licenses](#licenses)

## Installation

To set up the environment for this project, follow these steps:

1. **Extract the ZIP File**: 
   Extract the contents of this ZIP file to your local machine.

2. **Set Up a Python Environment**:
   Create a virtual environment and activate it (optional but recommended):
   
   ```bash
   python -m venv env
   source env/bin/activate   # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies**:
   Install the necessary dependencies for each section of the project using `pip`:

   For **TorchXRayVision**:
   ```bash
   pip install numpy matplotlib torch torchvision torchxrayvision
   ```

   For **Polyp Segmentation**:
   ```bash
   pip install opencv-python numpy lxml
   ```

## Usage

All the code required to run the models is contained within the Jupyter notebooks. To use the code, simply open the relevant notebook and run all the cells in order.

- **TorchXRayVision**: Open `torchxrayvision.ipynb` to generate chest X-ray segmentation masks.
  
- **SPADE-GAN**: Open `SPADE_GAN.ipynb` to train and test SPADE-GAN on your dataset.

- **StyleGAN**: Open `StyleGAN.ipynb` to train and test StyleGAN for synthetic image generation.

- **Polyp Segmentation Masks**: Open `Polyps Segmentation Masks.ipynb` to generate segmentation masks for polyps.

## Results

After running the models, you should expect to see:
- **SPADE-GAN**: Synthetic medical images generated based on input segmentation masks.
- **TorchXRayVision**: Segmentation masks for different anatomical structures in chest X-rays.
- **StyleGAN**: High-resolution synthetic medical images based on the dataset provided.
- **Image Dataset Complexity Metrics**: Open Image_Dataset_Complexity_Metrics_Anonymized.ipynb to calculate dataset complexity metrics such as entropy for your medical images.
## Datasets

This study uses the following publicly available datasets:

- **[ISIC-2018](https://challenge.isic-archive.com/data/)**: This dataset is used for the ISIC Skin Lesion Analysis challenge. The reference for the dataset is:
  - Codella, N. C., Rotemberg, V., Tschandl, P., et al. "Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2018 International Skin Imaging Collaboration (ISIC)." *arXiv preprint arXiv:1902.03368* (2018). [Link to dataset](https://challenge.isic-archive.com/data/)

- **[Chest X-Ray Images](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)**: This dataset contains X-ray images for pneumonia detection, provided by Kaggle. The reference is:
  - Kermany, D., Zhang, K., and Goldbaum, M. "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification." *Mendeley Data* (2018). [Link to dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

- **[Colonoscopy Polyp Detection and Classification](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/FCBUOR)**: This dataset is used for colonoscopy polyp detection. The reference is:
  - Borgli, H., Thambawita, V., Smedsrud, P. H., et al. "HyperKvasir, a comprehensive multi-class image and video dataset for gastrointestinal endoscopy." *Dataverse* (2021). [Link to dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/FCBUOR)

## References

1. [SPADE-GAN](https://github.com/NVlabs/SPADE)
2. [TorchXRayVision](https://github.com/mlmed/torchxrayvision)
3. [StyleGAN3](https://github.com/NVlabs/stylegan3)

## Acknowledgments

We would like to express our gratitude to the following for their contributions and support:

- **NVIDIA NVlabs** for providing open-source implementations of **SPADE-GAN** and **StyleGAN3** through their [GitHub repositories](https://github.com/NVlabs).
- **The developers of TorchXRayVision** for making their tools available through the [TorchXRayVision GitHub repository](https://github.com/mlmed/torchxrayvision).

## Licenses

This project contains code based on several open-source repositories, each under its own license:

1. **SPADE-GAN** by NVIDIA NVlabs is licensed under the [CC BY-NC 4.0 License](https://github.com/NVlabs/SPADE/blob/master/LICENSE.md).
   
2. **StyleGAN3** by NVIDIA NVlabs is licensed under the [CC BY-NC 4.0 License](https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt). T
   
3. **TorchXRayVision** is licensed under the [MIT License](https://github.com/mlmed/torchxrayvision/blob/master/LICENSE).
