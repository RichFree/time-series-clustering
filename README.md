# Time Series Clustering

This codebase uses image encoders (VICReg) to generate time series embeddings for
clustering.

## Installation

Install from the `requirements.txt`

## Dataset

You must download the `UCRArchive_2018` [[1]](#1) zipped dataset from
[here](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/). Then unzip in your downloaded directory and symlink it into the `data` directory.

```bash
# after downloading your UCRArchive_2018 dataset
cd ~/Downloads
unzip UCRArchive_2018.zip
cd <project root>
ln -s ~/Downloads/UCRArchive_2018 data/.
```

## Train and Benchmark

To train:

```python
python train.py
```

The checkpoint files are saved into the `checkpoints` folder.

To benchmark:

```python
python benchmark.py
```

To plot:

```python
python plot.py
```

use the `SELECT` variable to choose which dataset to plot the predicted clusters

## Acknowledgements

The UCR Time Series Classification Archive is the dataset that makes
benchmarking this method possible.

The VICReg training code implementation was borrowed from https://github.com/imbue-ai/self_supervised

## References

<a id="1">[1]</a> 
Hoang Anh Dau, Eamonn Keogh, Kaveh Kamgar, Chin-Chia Michael Yeh, Yan Zhu, Shaghayegh Gharghabi , Chotirat Ann Ratanamahatana, Yanping Chen, Bing Hu, Nurjahan Begum, Anthony Bagnall , Abdullah Mueen, Gustavo Batista, & Hexagon-ML (2019). The UCR Time Series Classification Archive. URL https://www.cs.ucr.edu/~eamonn/time_series_data_2018/ 
