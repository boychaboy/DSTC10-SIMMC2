PATH_TO_GOLD_RESPONSES=./gpt2_dst/data/simmc2_dials_dstc10_dev_target.txt
PATH_TO_GROUNDTRUTH_RETRIEVAL=../../data/



# <Subtask 4 Generation>
$ python tools/response_evaluation.py \
    --data_json_path={PATH_TO_GOLD_RESPONSES} \
    --model_response_path={PATH_TO_MODEL_RESPONSES} \
    --single_round_evaluation

# <Subtask 4 Retrieval>
$ python tools/retrieval_evaluation.py \
    --retrieval_json_path={PATH_TO_GROUNDTRUTH_RETRIEVAL} \
    --model_score_path={PATH_TO_MODEL_CANDIDATE_SCORES} \
    --single_round_evaluation    
