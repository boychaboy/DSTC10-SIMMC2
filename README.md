# The Second Situated Interactive MultiModal Conversations (SIMMC 2.0) Challenge 2021

Codes submitted for [SIMMC 2.0 challenge](https://github.com/facebookresearch/simmc2), Track 3 of [DSTC 10](https://dstc10.dstc.community)

## Brief summary
### Multi-task model with Finetuning End-to-End GPT-2
We use the baseline architecture of GPT-2 model and uses large model with beamsize of 2. We modified the retrieval method to use cosine similarity when measuring retrieval scores. 


## Run Models

1. **Preprocess** the datasets to reformat the data for GPT-2 input.

```
$ cd mm_dst
$ ./run_preprocess_gpt2.sh
```
2. **Train** the baseline model

```
$ ./run_train_gpt2.sh
```
3. **Generate** prediction for `devtest | teststd` data

```
$ ./run_generate_gpt2.sh
```
4. **Reformat** predictions for subtask4-generation & subtask4-retrieval

```
" for devtest
$ ./run_format_subtask4.sh
" for teststd
$ ./run_format_subtask4_teststd.sh
```
Evaluation reports are saved in the `/mm_dst/results` folder as JSON files.


## Devtest Results
We submit 2 different model predictions. Each model is trained with different hyperparameters and generated with different beam size. 
**MODEL1** is generated with beam size of 2
**MODEL2** is generated with beam size of 3


**MODEL1**  
- output dir : `model/mm_dst/gpt2_dst/results/devtest/model/`
- [download link](https://drive.google.com/drive/folders/1-aEUovt0Rj8mUvPAepawCb2b5xzXyH1-?usp=sharing)

**MODEL2** 
- output dir : `model/mm_dst/gpt2_dst/results/devtest/model_2/`  
- [download link](https://drive.google.com/drive/folders/1ukt22EeJtv0T0z2CbZBGPamkYglBXZ2Q?usp=sharing)

**Subtask #2: Multimodal Coreference Resolution**

| Baseline | Object F1 |
| :------: | :-------: |
| MODEL1   |   0.4657  |
| MODEL2   |   0.4697  |

**Subtask #3: Multimodal Dialog State Tracking**

| Baseline | Dialog Act F1 | Slot F1 | Request Slot F1 | Joint Accuracy |
| :------: | :-----------: | :-----: | :-------------: | :------------: |
| MODEL1   | 0.9613        | 0.8739  | 0.9385          | 0.5399         |
| MODEL2   | 0.9613        | 0.8825  | 0.9395          | 0.5474         |


**Subtask #4: Multimodal Dialog Response Generation** 

**Generation** 

| Baseline |      BLEU |
| :------: | :-------: |
| MODEL1   |   0.2838  |
| MODEL2   |   0.2723  |

**Retrieval**  

| Baseline |    MRR    |  R@1 | R@5 | R@10 | Mean Rank |
| :------: | :-------: | :---: | :-------: | :------: | :-------: |
| MODEL1   |   0.5164   |   0.4041   |   0.6402   |  0.732   |   11.91   |
| MODEL2   |   0.5005   |   0.3885   |   0.6239   |  0.7207  |   12.24   |


## Teststd Results
We submit 2 different model predictions for teststd

`MODEL1` output dir : `model/mm_dst/gpt2_dst/results/teststd/model/`  
`MODEL2` output dir : `model/mm_dst/gpt2_dst/results/teststd/model_2/`  

## License

SIMMC 2.0 is released under [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode), see [LICENSE](LICENSE) for details.
