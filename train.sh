#MODEL_DIR="models/roberta"
OUTPUT_DIR="models/roberta/weights"
TRAIN_PATH="dataset/src-train.txt"
EVAL_PATH="dataset/src-val.txt"
TOKENIZER_NAME="models/roberta"
CONFIG_NAME="models/roberta"

mkdir -p  $WEIGTHS_DIR

python3 model_config.py

# Arguments documentation: https://huggingface.co/transformers/master/_modules/transformers/training_args.html
python3 transformers/examples/run_language_modeling.py \
    --output_dir $OUTPUT_DIR \
    --model_type roberta \
    --mlm \
    --config_name $CONFIG_NAME \
    --tokenizer_name $TOKENIZER_NAME \
    --train_data_file $TRAIN_PATH \
    --eval_data_file $EVAL_PATH \
    --do_train \
    --do_eval \
    --num_train_epochs 1 \
    --overwrite_output_dir \
    --block_size 512 \
    --max_step 25 \
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
