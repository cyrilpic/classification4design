# Classification Algorithms for Engineering Design

This repository contains to code to evaluate different classification algorithms
on eight benchmark problems from engineering design. In particular, it evaluates
the novel pretrained classification model: [TabPFN](https://github.com/automl/TabPFN).

## Datasets and Results

The datasets used to evaluate the various classification algorithms are available
[here](https://doi.org/10.7910/DVN/ZRHXNY). Download the whole `data` folder and
put it at the root of this repository.

The datasets are located in `data/processed`, while the performance of the considered
classifiers are in `data/results`. The files are in Arrow format (parquet) and
are best read with `pandas`.

To recreate the plots of our paper, you can run the `plot.ipynb` notebook.

## Citation

If you use the datasets or the code for research purposes, you can cite our paper:

Cyril Picard and Faez Ahmed, Fast and Accurate Zero-Training Classification for Tabular Engineering Data, [arXiv:2401.06948](https://doi.org/10.48550/arXiv.2401.06948).