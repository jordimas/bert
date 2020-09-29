python3 /home/jordi/sc/transformers/examples/language-modeling/run_language_modeling.py run_language_model.py \
	--output_dir models/catalan-small-v1 \
	--model_type roberta \
	--mlm \
	--tokenizer_name models/catalan-tokenizer \
	--do_train \
	--learning_rate 1e-4 \
	--num_train_epochs 5 \
	--save_total_limit 2 \
	--save_steps 2000 \
	--per_gpu_train_batch_size 4 \
	--evaluate_during_training \
	--seed 42 \
	--train_data_file dataset/src-train.txt \
/
