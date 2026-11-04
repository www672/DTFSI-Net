# DTFSI-Net: Decomposition-Driven Temporal-Frequency-Semantic Interaction Network for Long-Term Time Series Forecasting 


## Introduction

DTFSI-Net is a Decomposition-Driven Temporal-Frequency-Semantic Interaction Network for long-term time series forecasting. Real-world time series mix long-term trends, periodic patterns and local variations, and naive cross-modal fusion between time series and text often fails to align semantic information with different temporal patterns and leads to representation redundancy. Our model decomposes trend and periodic components using moving average decomposition and a learnable cyclic queue, leverages text-generated gating for trend enhancement, and adopts a Fourier-based frequency branch with textual semantic modulation. An adaptive gate fuses time- and frequency-domain features, and cross-modal attention builds associations between numerical and textual representations. 

<p align="center">
<img src=".\images\FIGURE 2.png" width = "700" alt="" align=center />
<br><br>
</p>

## Environment Requirements

To get started, ensure you have Conda installed on your system and follow these steps to set up the environment:

```
conda create -n DTFSINet python=3.10
conda activate DTFSINet
pip install -r requirements.txt
```

## Data Preparation

* Pre-processed datasets can be downloaded from the following
  links, [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/e1ccfff39ad541908bae/)
  or [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing).
* Place the downloaded datasets into the `dataset/` folder, e.g. `dataset/ETTh1.csv`.

## Usage

1. Install the required dependencies.
2. Download data as above, and place them in the folder, `dataset/`.
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`,
   e.g. `./scripts/ETTh1.sh`. You might have to change permissions on the script files by running`chmod u+x scripts/*`.

## Acknowledgement

We extend our heartfelt appreciation to the following GitHub repositories for providing valuable code bases and datasets:

* https://github.com/thuml/iTransformer

* https://github.com/salesforce/ETSformer

* https://github.com/cure-lab/LTSF-Linear

* https://github.com/zhouhaoyi/Informer2020

* https://github.com/thuml/Autoformer

* https://github.com/MAZiqing/FEDformer

* https://github.com/ts-kim/RevIN

* https://github.com/timeseriesAI/tsai

