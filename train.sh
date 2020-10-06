# Model paths
MODEL_TYPE="roberta" #@param ["roberta", "bert"]
MODEL_DIR="models/roberta" #@param {type: "string"}
OUTPUT_DIR="models/roberta/output" #@param {type: "string"}
TRAIN_PATH="data/train.txt" #@param {type: "string"}
EVAL_PATH="data/dev.txt" #@param {type: "string"}

output_dir=$OUTPUT_DIR
model_type=$MODEL_TYPE
config_name=$MODEL_DIR
tokenizer_name=$MODEL_DIR
train_path=$TRAIN_PATH
eval_path=$EVAL_PATH
do_eval="--do_eval"
evaluate_during_training=""
line_by_line=""
should_continue=""
model_name_or_path=""

python model_config.py

python run_language_modeling.py \
    --output_dir $output_dir \
    --model_type $model_type \
    --mlm \
    --config_name $config_name \
    --tokenizer_name $tokenizer_name \
    $line_by_line \
    $should_continue \
    $model_name_or_path \
    --train_data_file $train_path \
    --eval_data_file $eval_path \
    --do_train \
    $do_eval \
    $evaluate_during_training \
    --overwrite_output_dir \
    --block_size 512 \
    --max_step 50000 \
    --warmup_steps 10 \
    --learning_rate 5e-5 \
    --per_gpu_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --max_grad_norm 100.0 \
    --save_total_limit 10 \
    --save_steps 10 \
    --logging_steps 2 \
    --seed 42
