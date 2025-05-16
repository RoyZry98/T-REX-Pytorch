timestamp=$(date '+%Y-%m-%d_%H:%M:%S')
model_name=llama2-13b
OUTPUTDIR=LOG/${timestamp}_${model_name}_trex_eval
mkdir -p $OUTPUTDIR

CUDA_VISIBLE_DEVICES=0 python chat.py \
    --moe_mode rank1_flex \
    --rank1_flex_rank_allocation 4 8 \
    --adapter_name_or_path ./output/adapter_path \
    --template_name llama2 \
    --model_name_or_path llama2-13b_model_path \
    --datasets "multirc" "mmlu" "boolq" "wic" "WinoGrande" "wsc" "anli" "piqa" "siqa" "rte" "copa" "openbookqa" "commonsense_qa" "hellaswag" 2>&1 | tee $OUTPUTDIR/log-eval-${model_name}_trex.log