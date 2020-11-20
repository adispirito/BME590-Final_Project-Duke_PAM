# BME590-Final_Project-Duke_PAM
By Anthony DiSpirito

This code implements a U-net style architecture using Tensorflow 2.0 and Keras to upsample undersampled PAM images using the Duke PAM Tensorflow Dataset.

In this repository we provide:
- Code to train your own model using our input data pipeline in `ModelTraining.ipynb`
- Code to show example images of the input data pipeline, downsampling methods, and model output results in `./examples`
- Code containing some potential future work using modified U-net's or GANS in `./future_work`
- Example results stored in `./figures`
- Model definitions stored in `./models`
- Helper functions stored in `./utils`
- Docker image Jupyterlab Configuration stored in `./jupyter`
- Docker image stored in `Dockerfile`

## Example outputs

### Uniform Downsampling
#### Input image - Uniform (3,3)
<img src="figures/saved_unet_uni_down_3_3_down_img.png" width="1024">

#### Output image - Uniform (3,3)
<img src="figures/saved_unet_uni_down_3_3_pred_img.png" width="1024">

#### Ground Truth image
<img src="figures/saved_unet_uni_down_3_3_full_img.png" width="1024">

### Random Downsampling
#### Input image - Random (89% Pixels Missing)
<img src="figures/saved_unet_rand_down_89p_down_img.png" width="1024">

#### Output image - Random (89% Pixels Missing)
<img src="figures/saved_unet_rand_down_89p_pred_img.png" width="1024">

#### Ground Truth image
<img src="figures/saved_unet_uni_down_3_3_full_img.png" width="1024">

## Instructions:
### Google Colab Example Usage:
https://drive.google.com/file/d/1ZzFE9X2xtd9cyceLBES-kO9YxJ0C08rV/view?usp=sharing

### Docker Instructions
```
docker build --rm -t adispirito/duke_pam:2.3.1-gpu .
docker run -it --rm --gpus all -u root -p 8888:8888 -v "${PWD}":/tf adispirito/duke_pam:2.3.1-gpu
```
