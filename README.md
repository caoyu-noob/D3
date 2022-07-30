# D3
### The implementation for ACL 2022 paper 

### A Model-Agnostic Data Manipulation Method for Persona-based Dialogue Generation

![Framework](https://github.com/caoyu-noob/D3/blob/main/framework.PNG)

---

### File structure
1. The main entrance to train the model is in  `train.py` in the root directory. We also provide some example shells
for running under different conditions.
2. The code related to our data manipulation method D3 is under `./data_manipulation`, where you can obtain augmented
data by using code under this directory.
3. `./attention_experiment` contains scripts for the attention experiments (like Appendix C.1 and C.4) in our paper
4. `./model` contains scripts for all other necessary parts to run experiment, including models, optimizer, data interface
and so on.

---

### Requirements
1. python == 3.7.0
2. torch==1.5.0
3. transformers==3.1.0
4. spacy==2.2.4
5. fairseq==0.9.0 (I downloaded the source code into the root directory)
6. sentencepiece==0.1.94

For evaluating the generated responses, you need to install `java-1.8.0`, `perl`, java-1.8.0, as well as
 perl library including XML::Twig, Sort::Naturally, String::Util (I use cpanm to install them on Linux). 
 
[METEOR](https://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz) is also needed for evaluating the quality
 of responses, please unzip it and put it under `./metrics/`
 
We also use [BERTScore](https://github.com/Tiiiger/bert_score) as a metric in our experiments, you may need to download
a proper BERT model for a successful evaluation.

---
## Run the code

To be honest, just applying step 3.Data Distillation can achieve a satisfactory performance. The step 4.data diversification
contribute less to the final results and is more complex.

## 1. Obtain PersonaChat Dataset

Obtain PersonaChat dataset via [ParlAI](https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks/personachat)
 or [our zipped version](https://drive.google.com/file/d/1zQVO5MuEy3wBUfZpM39uYmloD3-Rld4T/view) and put them into the `./datasets` directory.

## 2. Prepare models
At first, we have to get all trained models we need for data manipulation in experiments.
You need go to `./data_manipulation/prepare_model`.

##### 1) NLI model for evaluating persona consistency
You need to download [DialogueNLI dataset](https://wellecks.github.io/dialogue_nli/)
and put it under this directory. Also, download large size [RoBERTa MNLI model](https://huggingface.co/roberta-large-mnli)
and put it under this directory, renaming the document as `roberta_mnli/`.

Then you can train the NLI model using this dataset using script `train_nli_model.py`.

After obtain the trained best model, you need to renamed the file `best_model.bin` as `pytorch_model.bin` for the following 
use. Define the path that saves the trained NLI model for persona consistency as `PERSONA_NLI`.

##### 2) NLI model for evaluating coherence of dialogue history

Using the same RoBERTa MNLI model we used in 1 and `train_coherence_nli.py` to train it on the [InferConvAI2 dataset](https://github.com/nouhadziri/DialogEntailment).
It is a dialogue NLI dataset designed for evaluating the coherence of dialogue history. 

Save the obtained model, define the path containing the model as `COHERENCE_NLI`.

##### 3) BERT and GPT2 model used in data diversification

First use `extract_personas_and_responses.py` to extract persona and response texts into two json files.

Download the [bert-based-uncased model](https://huggingface.co/bert-base-uncased) and [gpt2-small model](https://huggingface.co/gpt2),
put them under the corresponding directories you like.
Then using `finetune_bert_and_gpt2.py` to fine tune BERT and GPT2 model on `personas.json`, obtaining BERT<sub>per</sub> and 
GPT2<sub>per</sub>, then fine tune GPT2 on `responses.json` to obtain GPT2<sub>res</sub>, editing the code to assign the model paths
of BERT and GPT2 you just defined before.

##### 4) Back translation model for dialogue history diversification

Got to directory `./BT`.

Download [WMT14 en-fr corpus](http://statmt.org/wmt14/translation-task.html#Download), and pre-processing it with 
BPE from sentencepiece using `preprocess.sh`, obtaining `sentence.bpe.model`.

Train en-fr and fr-en translation model using `train_en-fr.sh` and `train_fr-en.sh` under this directory and the average the last 5 models using 
`average_model.sh`. Define the obtained model checkpoints as `BT_EN-FR` and `BT-FR-EN`.

## 3. Data Distillation

Go to `./data_augmentation/data_distillation`.

Using `calculate_entailment.py` to obtained the predicted results given by the NLI model under `PERSONA_NLI` 
you obtained before. 

Then using `get_distilled_dataset.py` to obtain the distilled dataset using the previously logits given by the NLI model.
Assume that the obtain distilled data file is `DISTILL_DATA`.

## 4. Data diversification

##### 1) Obtain the Multi-GPT2 model for response align under new personas
At first you need to obtain a Multi-GPT2 model trained on the distilled samples. You can use the shell 
`train_multi_gpt2_distilled.sh` under the root directory. Set the training data as `DISTILL_DATA`
 according to the definitions of `config.py`

##### 2) Augment dialogue history
Then you need to augment dialogue history. Go to `./BT`, using `get_bt_input_file.py` to transform the distilled data 
`DISTILL_DATA` into the format for back translation. Then use `bpe_split.py` to pre-process the newly obtained txt file with BPE. 

Using `evaluate.sh` and `evaluate_back.sh` you can translate all utterance into French and then back to English.

Finally, using `recover.py` you can recover the txt file into its original distilled data format in a json file.

##### 3) Editing personas and align responses
Go to `./data_augmentation/data_diversification`. Using `generate_new_personas_and_edit_responses.py` you can obtain 
new personas as well as some samples with edited new responses if applicable.

Using `inference_multi_gpt2.sh` in the root directory you can get the predicted responses for the rest samples.

Using `get_augmented_scores.py` you can get the filter scores for each new sample.

Using `filter_augmented_data.py` you can get the filtered diversified samples along with the distilled one. They form
the augmented dataset used as an easy curriculum for training.

## 5. Train model

Put the obtained augmented dataset into `./datasets/augmented/` and then you can train two models using 
`train_seq2seq_D3.sh` and `train_gpt2_D3.sh`.