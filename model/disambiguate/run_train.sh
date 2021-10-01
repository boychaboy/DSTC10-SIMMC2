EPOCH=10
LR=5e-5
BATCH=64

CUDA_VISIBLE_DEVICE=$1 python train_model.py \
	--train_file="../../data/simmc2_disambiguate_dstc10_train.json" \
	--dev_file="../../data/simmc2_disambiguate_dstc10_dev.json" \
	--devtest_file="../../data/simmc2_disambiguate_dstc10_devtest.json" \
    --result_save_path="results/" \
	--use_gpu \
    --batch_size=8 \
    --learning_rate=2e-5 \
    --max_turns=5
