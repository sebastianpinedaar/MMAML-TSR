# MMAML-TSR
## Multi-modal meta-learning for time series regression

This repo contains the implementation of this [paper](https://arxiv.org/abs/2108.02842) which adapts MAML ([Finn et al., 2017](https://arxiv.org/pdf/1703.03400.pdf)) and MMAML ([Vuorio et al., 2019](https://arxiv.org/pdf/1910.13616.pdf)) to Time Series Regression. 

The multimodal-meta-learning is based on this [official implementation](https://github.com/shaohua0116/MMAML-Classification).

### Dependencies

* Python 3.7.0
* Pytorch 1.4.0
* Learn2learn 0.1.1

### Data

The code can be used  on two open datasets that need to be pre-processed before running MAML or MMAML. The data is available on:

* **Air Pollution Dataset**: [PM2.5 Data of Five Chinese Cities Data Set](https://archive.ics.uci.edu/ml/datasets/PM2.5+Data+of+Five+Chinese+Cities). 

* **Heart Rate Dataset**: [PPG-DaLiA Data Set](https://archive.ics.uci.edu/ml/datasets/PPG-DaLiA).

### Usage
1. **Download data** and create the following folder structure:

```shell
MMAML-TSR/
├── logs
	├── MAML_output/ 
	├── MMAML_output/
	...
├── data
├── code
	├── tools
	├── pre_processing
	├── models
	...


```


1. **Preprocess and generate .pickle file**

Change the paths to the raw data in the file `pre_processing/ts_dataset.py` accordingly.  Then run `pre_processing/dataset_creation.ipynb` to pickle the object with the transformed data. For a new dataset, a loading functionality should be created by taking our datasets as reference. Optionally, you can download the preprocessed data [HERE](https://www.dropbox.com/sh/yds6v1uok3bjydn/AAC5GRWw0F3clopRlk00Smvza?dl=0).

2. **Run MAML**

Assuming that the pickled files are in `data/`. Training with the default parameters on the Air Pollution Dataset works as:

```shell
cd code/
python run_MAML.py
```

To train on Heart-rate data:

```shell
cd code/
python run_MAML.py --dataset HR
```


3. **Run MMAML**

Assuming that the pickled files are in `data/`. Training with the default parameters on the Air Pollution Dataset works as:

```shell
cd code
python run_MMAML.py
```

To train on Heart-rate data:

```shell
cd code
python run_MMAML.py --dataset HR
```

### Cite us
If this repository is useful, please cite us as:

```
@inproceedings{arango2021multimodal,
  title={Multimodal meta-learning for time series regression},
  author={Arango, Sebastian Pineda and Heinrich, Felix and Madhusudhanan, Kiran and Schmidt-Thieme, Lars},
  booktitle={Advanced Analytics and Learning on Temporal Data: 6th ECML PKDD Workshop, AALTD 2021, Bilbao, Spain, September 13, 2021, Revised Selected Papers 6},
  pages={123--138},
  year={2021},
  organization={Springer}
}
```

### Contact
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/sebastianpinedaar/MMAML-TSR/issues).



