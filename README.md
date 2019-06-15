# SQLova
- SQLova is a neural semantic parser translating natural language utterance to SQL query. The name is originated from the name of our department:  **S**earch & **QLova** ([Search & Clova](https://clova.ai/ko/research/publications.html)).

### Authors
- [Wonseok Hwang](mailto:wonseok.hwang@navercorp.com), [Jinyeong Yim](mailto:jinyeong.yim@navercorp.com), [Seunghyun Park](mailto:seung.park@navercorp.com), and [Minjoon Seo](https://seominjoon.github.io).
- Affiliation: Clova AI Research, NAVER Corp., Seongnam, Korea.
- [Technical report](https://ssl.pstatic.net/static/clova/service/clova_ai/research/publications/SQLova.pdf).

### Abstract
- We present the new state-of-the-art semantic parsing model that translates a natural language (NL) utterance into a SQL query.
- The model is evaluated on [WikiSQL](https://github.com/salesforce/WikiSQL), a semantic parsing dataset consisting of 80,654 (NL, SQL) pairs over 24,241 tables from Wikipedia.
- We achieve **83.6%** logical form accuracy and **89.6%** execution accuracy on WikiSQL test set.

### The model in a nutshell
- [BERT](https://arxiv.org/abs/1810.04805) based table- and context-aware word-embedding. 
- The sequence-to-SQL model leveraging recent works ([Seq2SQL](https://arxiv.org/abs/1709.00103), [SQLNet](https://arxiv.org/abs/1711.04436)).
- [Execution-guided decoding](https://arxiv.org/abs/1807.03100) is applied in SQLova-EG.

### Results (Updated at Jan 12, 2019)
| **Model**   | Dev <br />logical form <br />accuracy | Dev<br />execution<br/> accuracy | Test<br /> logical form<br /> accuracy | Test<br /> execution<br /> accuracy |
| ----------- | ------------------------------------- | -------------------------------- | -------------------------------------- | ----------------------------------- |
| SQLova    | 81.6 (**+5.5**)^                      | 87.2 (**+3.2**)^                 | 80.7 (**+5.3**)^                       | 86.2 (**+2.5**)^                    |
| SQLova-EG | 84.2 (**+8.2**)*                      | 90.2 (**+3.0**)*                 | 83.6(**+8.2**)*                        | 89.6 (**+2.5**)*                    |

- ^: Compared to current [SOTA](https://github.com/salesforce/WikiSQL) models that do not use execution guided decoding.
- *: Compared to current [SOTA](https://github.com/salesforce/WikiSQL).
- The order of where conditions is ignored in measuring logical form accuracy in our model. 



### Source code
#### Requirements
- `python3.6` or higher.
- `PyTorch 0.4.0` or higher.
- `CUDA 9.0`
- Python libraries: `babel, matplotlib, defusedxml, tqdm`
- Example
    - Install [minicoda](https://conda.io/miniconda.html)
    - `conda install pytorch torchvision -c pytorch`
    - `conda install -c conda-forge records==0.5.2`
    - `conda install babel` 
    - `conda install matplotlib`
    - `conda install defusedxml`
    - `conda install tqdm`
- The code has been tested on Tesla M40 GPU running on Ubuntu 16.04.4 LTS.

#### Running code
- Type `python3 train.py --seed 1 --bS 16 --accumulate_gradients 2 --bert_type_abb uS --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_leng 222` on terminal.
    - `--seed 1`: Set the seed of random generator. The accuracies changes by few percent depending on `seed`.
    - `--bS 16`: Set the batch size by 16.
    - `--accumulate_gradients 2`: Make the effective batch size be `16 * 2 = 32`.
    - `--bert_type_abb uS`: Uncased-Base BERT model is used. Use `uL` to use Uncased-Large BERT.
    - `--fine_tune`: Train BERT. Without this, only the sequence-to-SQL module is trained.
    - `--lr 0.001`: Set the learning rate of the sequence-to-SQL module as 0.001. 
    - `--lr_bert 0.00001`: Set the learning rate of BERT module as 0.00001.
    - `--max_seq_leng 222`: Set the maximum number of input token lengths of BERT.     
- The model should show ~79% logical accuracy (lx) on dev set after ~12 hrs (~10 epochs). Higher accuracy can be obtained with longer training, by selecting different seed, by using Uncased Large BERT model, or by using execution guided decoding.
- Add `--EG` argument while running `train.py` to use execution guided decoding. 
- Whenever higher logical form accuracy calculated on the dev set, following three files are saved on current folder:
    - `model_best.pt`: the checkpoint of the the sequence-to-SQL module.
    - `model_bert_best.pt`: the checkpoint of the BERT module.
    - `results_dev.jsonl`: json file for official evaluation.
- `Shallow-Layer` and `Decoder-Layer` models can be trained similarly (`train_shallow_layer.py`, `train_decoder_layer.py`). 

#### Evaluation on WikiSQL DEV set
- To calculate logical form and execution accuracies on `dev` set using official evaluation script,
    - Download original [WikiSQL dataset](https://github.com/salesforce/WikiSQL).
    - tar xvf data.tar.bz2
    - Move them under `$HOME/data/WikiSQL-1.1/data`
    - Set path on `evaluation_ws.py`. This is the file where the path information has added on original `evaluation.py` script. Or you can use original [`evaluation.py`](https://github.com/salesforce/WikiSQL) by setting the path to the files by yourself.
    - Type `python3 evaluation_ws.py` on terminal.

#### Evaluation on WikiSQL TEST set
- Uncomment line 550-557 of `train.py` to load `test_loader` and `test_table`.
- One `test(...)` function, use `test_loader` and `test_table` instead of `dev_loader` and `dev_table`.
- Save the output of `test(...)` with `save_for_evaluation(...)` function.
- Evaluate with `evaluatoin_ws.py` as before.

#### Load pre-trained SQLova parameters.
- Pretrained SQLova model parameters are uploaded in [release](https://github.com/naver/sqlova/releases). To start from this, uncomment line 562-565 and set paths.

  
#### Code base 
- Pretrained BERT models were downloaded from [official repository](https://github.com/google-research/bert). 
- BERT code is from [huggingface-pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT).
- The sequence-to-SQL model is started from the source code of [SQLNet](https://github.com/xiaojunxu/SQLNet) and significantly re-written while maintaining the basic column-attention and sequence-to-set structure of the SQLNet.

#### Data
- The data is annotated by using `annotate_ws.py` which is based on [`annotate.py`](https://github.com/salesforce/WikiSQL) from WikiSQL repository. The tokens of natural language guery, and the start and end indices of where-conditions on natural language tokens are annotated.
- Pre-trained BERT parameters can be downloaded from BERT [official repository](https://github.com/google-research/bert) and can be coverted to `pt`file using following script. You need install both pytorch and tensorflow and change `BERT_BASE_DIR` to your data directory.

```sh
    cd sqlova
    export BERT_BASE_DIR=data/uncased_L-12_H-768_A-12
    python bert/convert_tf_checkpoint_to_pytorch.py \
        --tf_checkpoint_path $BERT_BASE_DIR/bert_model.ckpt \
        --bert_config_file    $BERT_BASE_DIR/bert_config.json \
        --pytorch_dump_path     $BERT_BASE_DIR/pytorch_model.bin 
```

- `bert/convert_tf_checkpoint_to_pytorch.py` is from the previous version of [huggingface-pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT), and current version of `pytorch-pretrained-BERT` is not compatible with the bert model used in this repo due to the difference in variable names (in LayerNorm). See [this](https://github.com/naver/sqlova/issues/1) for the detail.
- For the convenience, the annotated WikiSQL data and the PyTorch-converted pre-trained BERT parameters are available at [here](https://drive.google.com/file/d/1iJvsf38f16el58H4NPINQ7uzal5-V4v4/view?usp=sharing).

### License
```
Copyright 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
