# ABME (ICCV2021)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/asymmetric-bilateral-motion-estimation-for/video-frame-interpolation-on-vimeo90k)](https://paperswithcode.com/sota/video-frame-interpolation-on-vimeo90k?p=asymmetric-bilateral-motion-estimation-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/asymmetric-bilateral-motion-estimation-for/video-frame-interpolation-on-x4k1000fps)](https://paperswithcode.com/sota/video-frame-interpolation-on-x4k1000fps?p=asymmetric-bilateral-motion-estimation-for)


Junheum Park,
Chul Lee,
and Chang-Su Kim

Official PyTorch Code for **"Asymmetric Bilateral Motion Estimation for Video Frame Interpolation"** [[paper]](https://arxiv.org/abs/2108.06815)

### Requirements
- PyTorch 1.7
- CUDA 11.0
- CuDNN 8.0.5
- python 3.8

### Installation
Create conda environment:
```bash
    $ conda create -n ABME python=3.8 anaconda
    $ conda activate ABME
    $ pip install opencv-python
    $ conda install pytorch==1.7 torchvision cudatoolkit=11.0 -c pytorch
```
Download repository:
```bash
    $ git clone https://github.com/JunHeum/ABME.git
```
Download [pre-trained model](https://drive.google.com/u/0/uc?id=1fRLxZ0rYjto2yI1nHuUQ1-OsNkYqq-mL&export=download) parameters:
```bash
    $ unzip ABME_Weights.zip
```
Check your `nvcc` version:
```bash
    $ nvcc --version
```
- To install correlation layer, you should match your `nvcc` version with cudatoolkit version of your conda environment. [[nvcc_setting]](https://github.com/JunHeum/ABME/blob/main/correlation_package/nvcc%20setting.md)

Install correlation layer:
```bash
    $ cd correlation_package
    $ python setup.py install
```
### Quick Usage
Generate an intermediate frame on your pair of frames:
```bash
    $ python run.py --first images/im1.png --second images/im3.png --output images/im2.png
```
### Test
1. Download the datasets.
2. Copy the path of the test dataset. (e.g., `/hdd/vimeo_interp_test`)
3. Parse this path into the `--dataset_root` argument.
4. (optional) You can ignore the `--is_save`. But, it yields a slightly different performance than evaluation on saved images.
```bash
    $ python test.py --name ABME --is_save --Dataset ucf101 --dataset_root /where/is/your/ucf101_dataset/path
    $ python test.py --name ABME --is_save --Dataset vimeo --dataset_root /where/is/your/vimeo_dataset/path
    $ python test.py --name ABME --is_save --Dataset SNU-FILM-all --dataset_root /where/is/your/FILM_dataset/path
    $ python test.py --name ABME --is_save --Dataset Xiph_HD --dataset_root /where/is/your/Xiph_dataset/path
    $ python test.py --name ABME --is_save --Dataset X4K1000FPS --dataset_root /where/is/your/X4K1000FPS_dataset/path
```
### Experimental Results
We provide interpolated frames on test datasets for fast comparison or users with limited GPU memory. Especially, the test on X4K1000FPS requires at least 20GB of GPU memory.
- [UCF101](https://drive.google.com/uc?id=1xrC0jP1XfusMMMuUr87rhVkQqKrUGePk&export=download) 
- [Vimeo90K](https://drive.google.com/uc?id=1gMlkTgW5G17JUbrWhMqYjyz8L-_i6-kk&export=download)
- [SNU-FILM](https://drive.google.com/uc?id=1VloPEOQj-uKoS0tuORo5k9uAdhfOq0ys&export=download)
- [Xiph](https://drive.google.com/u/0/uc?id=163mb5xrpFN8gN7WvJfDSzuPBHTGiY19H&export=download)
- [X4K1000FPS](https://drive.google.com/uc?id=1OXyPw8_4zNWVbcd8k4Za6NDRtdTYgzbo&export=download)

![Table](/figures/Table.png "Table")
### Train
We plan to share train codes soon!
### Citation
Please cite the following paper if you feel this repository useful.
```bibtex
    @inproceedings{park2021ABME,
        author    = {Park, Junheum and Lee, Chul and Kim, Chang-Su}, 
        title     = {Asymmetric Bilateral Motion Estimation for Video Frame Interpolation}, 
        booktitle = {International Conference on Computer Vision},
        year      = {2021}
    }
```
### License
See [MIT License](https://github.com/JunHeum/ABME/blob/master/LICENSE)

