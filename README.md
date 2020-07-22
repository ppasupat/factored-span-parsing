# Span-based Hierarchical Semantic Parsing for Task-Oriented Dialog

This is the codebase for the paper
> Panupong Pasupat, Sonal Gupta, Karishma Mandyam, Rushin Shah, Mike Lewis, Luke Zettlemoyer.  
> [**Span-based Hierarchical Semantic Parsing for Task-Oriented Dialog**](https://www.aclweb.org/anthology/D19-1163/).  
> EMNLP, 2019

## Dependencies

* Python 3.x
* PyTorch 1.x
* [PyYAML](https://pypi.org/project/PyYAML/)
* [tqdm](https://github.com/tqdm/tqdm/)
* Download the processed GloVe vectors [here](https://nlp.stanford.edu/projects/phrasenode/processed-glove.zip)
  and extract the files to `data/glove/`.

## Usage

* Training with the basic model (no edge scores) on debug data:
  ```bash
  ./main.py train configs/base.yml configs/data/artificial-chain.yml configs/model/span-node.yml
  ```
  This will train the model on the **training** data and evaluate on the **development** data.
  The results will be saved to `out/___.exec/` where `___` is some number. The results contain
  the final config file, saved models, and predictions.

* To use the real training data (TOP dataset), change the `data` config to `configs/data/top-dataset.yml`.

* To use edge scores, change the `model` config to `configs/model/span-edge.yml`.

* The `out/__.exec/__.meta` metadata file stores the best accuracy and the epoch for that accuracy.
  To dump the metadata, use `./dump-meta.py out/__.exec/__.meta`.
  
* To dump the predictions of a trained model in `out/42.exec/14.model` on the **test** data:
  ```bash
  ./main.py test out/42.exec/config.json -l out/42.exec/14
  ```
