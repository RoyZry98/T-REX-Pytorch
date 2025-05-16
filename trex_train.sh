CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port $((40000 + RANDOM % 10001)) \
--nproc_per_node=8 train.py \
--train_args_file "./train_args/llama2-13b-trex.json" \
--PDdataset \
--cluster_file ./datasets/task_embs_32cluster_nomic-embed-text-v1.pt \
--moe_mode 'rank1_flex' \
--rank1_flex_rank_allocation 4 8

