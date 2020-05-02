[![Build Status](https://travis-ci.org/herilalaina/mosaic.svg?branch=master)](https://travis-ci.org/herilalaina/mosaic)

# Mosaic
Mosaic is a Python library for pipeline optimization. This library implements Monte-Carlo Tree Search algorithm in order to find optimal pipeline.

[Documentation](https://herilalaina.github.io/mosaic/)

### Installation

Requirements
* Python >= 3.5.6
* pygraphviz: necessary to generate dot image files (optional)
```commandline
conda install graphviz
pip install pygraphviz
```

Install via Github:
```bash
pip install git+https://github.com/herilalaina/mosaic
```


### Example of usage: machine learning
A simple example of using `mosaic` to configure machine
learning pipeline made with PCA and SVM classifier.

```bash
python examples/machine_learning.py
```

### Citation
If you are using `mosaic` in a academic presentation, we would appreciate citation
```
@inproceedings{ijcai2019-457,
  title     = {Automated Machine Learning with Monte-Carlo Tree Search},
  author    = {Rakotoarison, Herilalaina and Schoenauer, Marc and Sebag, Michèle},
  booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on
               Artificial Intelligence, {IJCAI-19}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  pages     = {3296--3303},
  year      = {2019},
  month     = {7},
  doi       = {10.24963/ijcai.2019/457},
  url       = {https://doi.org/10.24963/ijcai.2019/457},
}
```
