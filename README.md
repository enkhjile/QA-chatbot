# QA-chatbot
Answers to questions based on context. Used [Reformer](https://arxiv.org/abs/2001.04451) architecture.

## Install required packages
### 1. mecab-ipadic-neologd
##### Step1. Install mecab
Ubuntu
```
sudo apt install mecab
sudo apt install libmecab-dev
sudo apt install mecab-ipadic-utf8
```
MacOS
```
brew install mecab mecab-ipadic
```
Check if its working
```
echo 特急はくたか | mecab

特急	名詞,一般,*,*,*,*,特急,トッキュウ,トッキュー
は	助詞,係助詞,*,*,*,*,は,ハ,ワ
く	動詞,自立,*,*,カ変・クル,体言接続特殊２,くる,ク,ク
た	助動詞,*,*,*,特殊・タ,基本形,た,タ,タ
か	助詞,副助詞／並立助詞／終助詞,*,*,*,*,か,カ,カ
EOS
```
##### Step2. Update Dictionary
```
git clone https://github.com/neologd/mecab-ipadic-neologd.git
cd mecab-ipadic-neologd
sudo bin/install-mecab-ipadic-neologd
```
Check directory path of ipadic-neologd dictionary
```
echo `mecab-config --dicdir`"/mecab-ipadic-neologd"

/usr/local/lib/mecab/dic/mecab-ipadic-neologd
```
Usage
```
echo 特急はくたか | mecab -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd

特急	名詞,一般,*,*,*,*,特急,トッキュウ,トッキュー
はくたか	名詞,固有名詞,一般,*,*,*,はくたか,ハクタカ,ハクタカ
EOS
```

### 2. Python packages
Install python packages
```
pip install -r requirements.txt
```

## Usage
Change mecab_dict in **./models/config.yml** file to mecab-ipadic-neologd dictionary path installed on you machine.  
To check mecab-ipadic-neologd dictionary path

```
echo `mecab-config --dicdir`"/mecab-ipadic-neologd"
```
**./models/config.yml**
```
model:
  num_tokens: 32000
  dim: 512
  depth: 12
  max_seq_len: 256
  heads: 8
  casual: True
  return_embeddings: True
  num_label: 2

tokenizer:
  max_len: 256
  mecab_dict: <mecab-ipadic-neologd dictionary path installed on you machine>
  vocab: ./models/mecab_vocab.txt

predict:
  n_best_size: 1
  doc_stride: 64
  max_query_length: 64
  max_answer_length: 30
```

Talk with the BOT.
```
python main.py </path/to/config> --model_path=</path/to/model>
```
* </path/to/config>: **./models/config.yml**
* </path/to/model>: **./models/model_state_dict_1500.pt**  
Download pretrained model from [here](https://drive.google.com/drive/folders/1jJjVI-vmzd-9wQ6iArJFpw6DibNEIAZd?usp=sharing) and place it in **./models** directory.

Run Demo jupyter notebook (self explatory)
```
jupyter notebook demo.ipynb
```
