# Disentangled Contrastive Learning for Learning Robust Textual Representations (DCL)

Code for the CICAI paper [Disentangled Contrastive Learning for Learning Robust Textual Representations](https://arxiv.org/pdf/2104.04907.pdf).

## Requirements

* To install basic requirements:

```
pip install requirements.txt
```

## Datasets

* SQuAD 1.1, you can find the dataset on https://rajpurkar.github.io/SQuAD-explorer/
* GLUE , you can find the dataset on https://gluebenchmark.com/tasks

## Train DCL

To train the DCL model with given dataset, run `dcl_main.py`

```shell
./scripts/run.sh
```

## fine-tune GLUE

After the DCL model trained, fine-tune it on the GLUE dataset. Also, we could use the script to fine-tune Bert model.

```python
./scripts/run_dcl_glue.sh
```
Also, we can use the script to fine-tune the `Bert+da` model refered in the paper.

```python
./scripts/run_da_glue.sh
```

Also, we can use the script to fine-tune the `Bert` model refered in the experiments introduced in the paper.

```python
./scripts/run_raw_glue.sh
```

## fine-tune SQuAD1.1

After the DCL model trained, fine-tune it on the SQuAD1.1 dataset. Also, we could use the script to fine-tune Bert model.

```python
./scripts/run_squad.sh
```

## Data augmentation

To build the enhanced dataset for training `Bert+da` model, we Incorporate the `da` processing module into the Class `GlueDaDataset` in the da_utils.py 

```python
./da_utils.py   # for checklist data augmentation
```

To build the enhanced dataset for training `Bert+attack` model.

```python
python ./utils/filter_correct.py   # for filter correct data 
--input_file ../glue_data/CoLA
--task_name CoLA
--model_path bert-base-uncased
--output_file ./adv_data/CoLA 

python ./utils/attack.py           # for openattack data augmentation
--input_file ./adv_data/CoLA/train_correct.txt
--task_name CoLA
--model_path bert-base-uncased
--output_file ./adv_data/CoLA
--attacker pw          
```

# Citation
If you use the code, please cite the following paper:


```bibtex
@inproceedings{DBLP:conf/cicai/ChenXBYDZC21,
  author    = {Xiang Chen and
               Xin Xie and
               Zhen Bi and
               Hongbin Ye and
               Shumin Deng and
               Ningyu Zhang and
               Huajun Chen},
  editor    = {Lu Fang and
               Yiran Chen and
               Guangtao Zhai and
               Z. Jane Wang and
               Ruiping Wang and
               Weisheng Dong},
  title     = {Disentangled Contrastive Learning for Learning Robust Textual Representations},
  booktitle = {Artificial Intelligence - First {CAAI} International Conference, {CICAI}
               2021, Hangzhou, China, June 5-6, 2021, Proceedings, Part {II}},
  series    = {Lecture Notes in Computer Science},
  volume    = {13070},
  pages     = {215--226},
  publisher = {Springer},
  year      = {2021},
  url       = {https://doi.org/10.1007/978-3-030-93049-3\_18},
  doi       = {10.1007/978-3-030-93049-3\_18},
  timestamp = {Fri, 14 Jan 2022 09:56:37 +0100},
  biburl    = {https://dblp.org/rec/conf/cicai/ChenXBYDZC21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


