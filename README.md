# MMAML-TSR
## Multi-modal meta-learning for time series regression

This repo contains the implementation of the paper (\*) which adapts MAML ([Finn et al., 2017](https://arxiv.org/pdf/1703.03400.pdf)) and MMAML ([Vuorio et al., 2019](https://arxiv.org/pdf/1910.13616.pdf)) to Time Series Regression. 

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

Change the paths to the raw data in the file `pre_processing/ts_dataset.py` accordingly.  Then run `pre_processing/dataset_creation.ipynb` to pickle the object with the transformed data. For a new dataset, a loading functionality should be created by taking our datasets as reference.

1. **Run MAML**

Assuming that the pickled files are in `data/`. Training with the default parameters on the Air Pollution Dataset works as:

```shell
python run_MAML.py
```

To train on Heart-rate data:

```shell
python run_MAML.py --dataset HR
```


1. **Run MMAML**

Assuming that the pickled files are in `data/`. Training with the default parameters on the Air Pollution Dataset works as:

```shell
python run_MMAML.py
```

To train on Heart-rate data:

```shell
python run_MMAML.py --dataset HR
```

### Contact
To ask questions or report issues, please open an issue on the [issues tracker] (https://github.com/sebastianpinedaar/MMAML-TSR/issues).



