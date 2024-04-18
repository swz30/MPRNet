

# Multi-Stage Progressive Image Restoration (CVPR 2021)

[Syed Waqas Zamir](https://scholar.google.ae/citations?hl=en&user=POoai-QAAAAJ), [Aditya Arora](https://adityac8.github.io/), [Salman Khan](https://salman-h-khan.github.io/), [Munawar Hayat](https://scholar.google.com/citations?user=Mx8MbWYAAAAJ&hl=en), [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en), [Ming-Hsuan Yang](https://scholar.google.com/citations?user=p9-ohHsAAAAJ&hl=en), and [Ling Shao](https://scholar.google.com/citations?user=z84rLjoAAAAJ&hl=en)

[![paper](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://arxiv.org/abs/2102.02808)
[![supplement](https://img.shields.io/badge/Supplementary-Material-B85252)](https://drive.google.com/file/d/1mbfljawUuFUQN9V5g0Rmw1UdauJdckCu/view?usp=sharing)
[![video](https://img.shields.io/badge/Video-Presentation-F9D371)](https://www.youtube.com/watch?v=0SMTPiLw5Vw)
[![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)](https://drive.google.com/file/d/1-L43wj-VTppkrR9AL6cPBJI2RJi3Hc_z/view?usp=sharing)

<hr />

### News

We are happy to see that our work has inspired the **Winning Solutions in NTIRE 2021 challenges**:
- [Dual-pixel Defocus Deblurring Challenge](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Abuolaim_NTIRE_2021_Challenge_for_Defocus_Deblurring_Using_Dual-Pixel_Images_Methods_CVPRW_2021_paper.pdf) -- MRNet: Multi Refinement Network for Dual-pixel Images Defocus Deblurring
- [Image Deblurring Challenge](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Nah_NTIRE_2021_Challenge_on_Image_Deblurring_CVPRW_2021_paper.pdf) -- HINet: Half Instance Normalization Network for Image Restoration

<hr />

> **Abstract:** *Image restoration tasks demand a complex balance between spatial details and high-level contextualized information while recovering images. In this paper, we propose a novel synergistic design that can optimally balance these competing goals. Our main proposal is a multi-stage architecture, that progressively learns restoration functions for the degraded inputs, thereby breaking down the overall recovery process into more manageable steps. Specifically, our model first learns the contextualized features using encoder-decoder architectures and later combines them with a high-resolution branch that retains local information. At each stage, we introduce a novel per-pixel adaptive design that leverages in-situ supervised attention to reweight the local features. A key ingredient in such a multi-stage architecture is the information exchange between different stages. To this end, we propose a two-faceted approach where the information is not only exchanged sequentially from early to late stages, but lateral connections between feature processing blocks also exist to avoid any loss of information. The resulting tightly interlinked multi-stage architecture, named as MPRNet, delivers strong performance gains on ten datasets across a range of tasks including image deraining, deblurring, and denoising. For example, on the Rain100L, GoPro and DND datasets, we obtain PSNR gains of 4 dB, 0.81 dB and 0.21 dB, respectively, compared to the state-of-the-art.* 

## Network Architecture
<table>
  <tr>
    <td> <img src = "https://i.imgur.com/69c0pQv.png" width="500"> </td>
    <td> <img src = "https://i.imgur.com/JJAKXOi.png" width="400"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Overall Framework of MPRNet</b></p></td>
    <td><p align="center"> <b>Supervised Attention Module (SAM)</b></p></td>
  </tr>
</table>

## Installation
The model is built in PyTorch 1.1.0 and tested on Ubuntu 16.04 environment (Python3.7, CUDA9.0, cuDNN7.5).

For installing, follow these intructions
```
conda create -n pytorch1 python=3.7
conda activate pytorch1
conda install pytorch=1.1 torchvision=0.3 cudatoolkit=9.0 -c pytorch
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
```

Install warmup scheduler

```
cd pytorch-gradual-warmup-lr; python setup.py install; cd ..
```

## Quick Run

To test the pre-trained models of [Deblurring](https://drive.google.com/file/d/1QwQUVbk6YVOJViCsOKYNykCsdJSVGRtb/view?usp=sharing), [Deraining](https://drive.google.com/file/d/1O3WEJbcat7eTY6doXWeorAbQ1l_WmMnM/view?usp=sharing), [Denoising](https://drive.google.com/file/d/1LODPt9kYmxwU98g96UrRA0_Eh5HYcsRw/view?usp=sharing) on your own images, run 
```
python demo.py --task Task_Name --input_dir path_to_images --result_dir save_images_here
```
Here is an example to perform Deblurring:
```
python demo.py --task Deblurring --input_dir ./samples/input/ --result_dir ./samples/output/
```

## Training and Evaluation

Training and Testing codes for deblurring, deraining and denoising are provided in their respective directories.

## Results
Experiments are performed for different image processing tasks including, image deblurring, image deraining and image denoising. Images produced by MPRNet can be downloaded from Google Drive links: [Deblurring](https://drive.google.com/drive/folders/12jgrGdIh_lfiSsXyo-QicQuZYcLXp9rP?usp=sharing), [Deraining](https://drive.google.com/drive/folders/1IpF_jCGBhqsXN4f1vBNQ6DGpr7Pk6LdO?usp=sharing), and [Denoising](https://drive.google.com/drive/folders/1usbZKuYg8c7UrUml2bdZSbuxh_JrHW67?usp=sharing).

<details>
  <summary> <strong>Image Deblurring</strong> (click to expand) </summary>
<table>
  <tr>
    <td> <img src = "https://i.imgur.com/UIwmY13.png" width="450"> </td>
    <td> <img src = "https://i.imgur.com/ecSlcEo.png" width="450"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Deblurring on Synthetic Datasets.</b></p></td>
    <td><p align="center"><b>Deblurring on Real Dataset.</b></p></td>
  </tr>
</table></details>

<details>
  <summary> <strong>Image Deraining</strong> (click to expand) </summary>
<img src = "https://i.imgur.com/YVXWRJT.png" width="900"></details>

<details>
  <summary> <strong>Image Denoising</strong> (click to expand) </summary>
<p align="center"> <img src = "https://i.imgur.com/Wssu6Xu.png" width="450"> </p></details>

## Citation
If you use MPRNet, please consider citing:

    @inproceedings{Zamir2021MPRNet,
        title={Multi-Stage Progressive Image Restoration},
        author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat
                and Fahad Shahbaz Khan and Ming-Hsuan Yang and Ling Shao},
        booktitle={CVPR},
        year={2021}
    }

## Contact
Should you have any question, please contact waqas.zamir@inceptioniai.org

## Our Related Works
- Learning Enriched Features for Fast Image Restoration and Enhancement, TPAMI 2022. [Paper](https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/) | [Code](https://github.com/swz30/MIRNetv2)
- Restormer: Efficient Transformer for High-Resolution Image Restoration, CVPR 2022. [Paper](https://arxiv.org/abs/2111.09881) | [Code](https://github.com/swz30/Restormer)
- Learning Enriched Features for Real Image Restoration and Enhancement, ECCV 2020. [Paper](https://arxiv.org/abs/2003.06792) | [Code](https://github.com/swz30/MIRNet)
- CycleISP: Real Image Restoration via Improved Data Synthesis, CVPR 2020. [Paper](https://arxiv.org/abs/2003.07761) | [Code](https://github.com/swz30/CycleISP)
