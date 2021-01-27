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

`<+-- _config.yml \\
+-- _drafts>`

`<|   +-- begin-with-the-crazy-ideas.textile>`
`<|   +-- on-simplicity-in-technology.markdown>`
`<+-- _includes>`
`<|   +-- footer.html>`
`<|   +-- header.html>`
`<+-- _layouts>`
`<|   +-- default.html>`
`<|   +-- post.html>`


1. **Preprocess and generate .pickle file**
1. **Run MAML**
1. **Run MMAML**
### Contact


