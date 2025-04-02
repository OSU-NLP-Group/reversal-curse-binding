DATASET=data/$1/
WEIGHT_DECAY=$2
MAX_LENGTH=$5
GPU=$6

OUTPUT_DIR=<your_save_dir>/${1}_${2}_${3}_${4}_jepa

CUDA_VISIBLE_DEVICES=$GPU python main.py \
--data_dir $DATASET \
--n_layer $3 \
--weight_decay $WEIGHT_DECAY \
--max_length $MAX_LENGTH \
--output_dir $OUTPUT_DIR \
--train_batch_size 1024 \
--eval_batch_size 1024 \
--learning_rate 1e-4 \
--gradient_accumulation_steps 1 \
--save_steps 50000 \
--save_step_dense 30000 \
--save_step_dense_interval 5000 \
--max_steps 3000000 \
--scheduler constant_schedule_with_warmup \
--evaluate_during_training \
--init_weights \
--add_tokens \
--model_type gpt2 \
--model_name gpt2 \
--wernicke_broca \
--num_wernicke_layer $4 \
--num_broca_layer 0 \
--fp16 \
--fresh_tokenizer \
--eval_full_negatives