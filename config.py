from dataclasses import dataclass, field

@dataclass
class ModelArgs:
    # optimizer and scheduler
    data_dir: str = None
    data_dir_parent: str = "<your_parent_dir>"
    adam_betas: tuple = field(default_factory=lambda: (0.9, 0.999))
    adam_epsilon: float = 1e-8
    adafactor_beta1: float = None
    adafactor_clip_threshold: float = 1.0
    adafactor_decay_rate: float = -0.8
    adafactor_eps: tuple = field(default_factory=lambda: (1e-30, 1e-3))
    adafactor_relative_step: bool = False
    adafactor_scale_parameter: bool = False
    adafactor_warmup_init: bool = False
    cosine_schedule_num_cycles: float = 0.5
    scheduler: str = "constant_schedule_with_warmup"
    warmup_ratio: float = 0.06
    warmup_steps: int = 2000
    # training and eval setting
    dataloader_num_workers: int = 0
    train_batch_size: int = 512
    eval_batch_size: int = 512
    evaluate_during_training: bool = False
    evaluate_during_training_silent: bool = True
    evaluate_each_epoch: bool = False
    predict_during_training: bool = False
    fp16: bool = False
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    local_rank: int = -1
    manual_seed: int = 42
    max_grad_norm: float = 1.0
    max_length: int = 128
    multiprocessing_chunksize: int = -1
    eval_full_negatives: bool = False
    n_gpu: int = 1
    first_mlp_width: int = None
    disable_dropout: bool = False
    memory_block: bool = False
    memory_block_top_k: int = None
    init_wernicke_mlp_keys: bool = False
    memory_block_value_only: bool = False
    wb_lmloss: bool = False
    dont_save_models: bool = False
    num_train_epochs: int = 20
    overwrite_output_dir: bool = False
    save_model_every_epoch: bool = False
    save_optimizer_and_scheduler: bool = True
    save_steps: int = 50000
    save_step_dense: int = -1
    save_step_dense_interval: int = -1
    silent: bool = False
    skip_special_tokens: bool = True
    weight_decay: float = 0.1
    max_steps: int=1500000
    # lm/memory config
    n_layer: int = None
    n_embd: int = None
    n_head: int = None
    untie_word_embeddings: bool=False
    wernicke_broca: bool = False
    num_wernicke_layer: int = 3
    num_broca_layer: int = 3
    max_patch_length: int = 3    # number of tokens per patch
    fresh_tokenizer: bool = False
    concept_ln_type: str = "variance_only"
    # DDP
    local_rank: int = -1
    rank: int = -1
    gpu: int = -1
    world_size: int = -1
    dist_url: str = 'env://'
    dist_backend: str = 'nccl'

    def update_from_dict(self, new_values):
        assert type(new_values) == dict
        for key, value in new_values.items():
            # print(key, value)
            assert hasattr(self, key)
            setattr(self, key, value)