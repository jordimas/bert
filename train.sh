MODEL_DIR="models/roberta"
OUTPUT_DIR="models/roberta/output"

python3 model_config.py

# Arguments documentation: https://huggingface.co/transformers/master/_modules/transformers/training_args.html
CUDA_LAUNCH_BLOCKING=1 python3 transformers/examples/run_language_modeling.py \
    --output_dir $OUTPUT_DIR \
    --model_type roberta \
    --mlm \
    --train_data_file dataset/src-train.txt \
    --eval_data_file dataset/src-eval.txt \
    --config_name $MODEL_DIR \
    --tokenizer_name $MODEL_DIR \
    --do_train \
    --line_by_line \
    --overwrite_output_dir \
    --do_eval \
    --block_size 256 \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
    --save_total_limit 2 \
    --save_steps 2000 \
    --logging_steps 500 \
    --per_gpu_eval_batch_size 32 \
    --per_gpu_train_batch_size 32 \
    --evaluate_during_training
