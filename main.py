import json
import logging
import math
import os

import builtins
import random
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path
import functools
import pickle
import fire

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import read_json, SimpleDataset

import torch.multiprocessing as mp
from tqdm.auto import tqdm, trange
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

from config import ModelArgs

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, ["bias", "LayerNorm.weight", "ln"]),
}

logger = logging.getLogger(__name__)

BAN_POS_ENCODING = True

class Seq2SeqModule(nn.Module):
    """
    Just a wrapper.
    """
    def __init__(
        self,
        language_model,
    ):
        super(Seq2SeqModule, self).__init__()
        self.lm = language_model

    def forward(self, **kwargs):
        return self.lm(**kwargs)

    def save_pretrained(self, output_dir):
        self.lm.save_pretrained(output_dir)
        self.lm.config.save_pretrained(output_dir)
        if not (self.lm.concept_ln is None):
            torch.save(self.lm.concept_ln, os.path.join(output_dir, "concept_ln.pth"))
        if not (self.lm.broca_query is None):
            torch.save(self.lm.broca_query, os.path.join(output_dir, "broca_query.pth"))
        if type(self.lm.transformer.h[0]).__name__ == 'MemoryBlock':
            torch.save(self.lm.transformer.h[0], os.path.join(output_dir, "memory_block.pth"))

class Seq2SeqModel:
    def __init__(
        self,
        model_type,
        model_name,
        args=None,
        use_cuda=True,
        init_weights=False,
        vocab_file=None,
    ):
        print("numpy version:", np.__version__)
        print("torch version:", torch.__version__)
        print("transformers version:", transformers.__version__)

        ### load & update all general args
        self.args = ModelArgs()
        self.args.update_from_dict(args)

        ### GPU & distributed training setup
        self.args.n_gpu = torch.cuda.device_count()
        print("local gpu count:", self.args.n_gpu)
        if "WORLD_SIZE" in os.environ:
            self.args.world_size = int(os.environ["WORLD_SIZE"])
        self.distributed = self.args.world_size > 1
        if self.distributed:
            print("***In distributed mode, world_size:{}***".format(self.args.world_size))
            assert use_cuda
            if self.args.local_rank != -1:  # for torch.distributed.launch
                print("provided local_rank is {}. Setting rank and gpu both to be the same.".format(self.args.local_rank))
                self.args.rank = self.args.local_rank
                self.args.gpu = self.args.local_rank
            elif 'SLURM_PROCID' in os.environ:  # for slurm scheduler
                self.args.rank = int(os.environ['SLURM_PROCID'])
                self.args.gpu = self.args.rank % self.args.n_gpu
                print("provided local_rank is -1. Setting rank and gpu with SLURM_PROCID. Rank:{}, gpu:{}"
                      .format(self.args.rank, self.args.gpu))
            dist.init_process_group(backend=self.args.dist_backend, init_method=self.args.dist_url, world_size=self.args.world_size, rank=self.args.rank)
            assert self.args.rank >= 0
        else:
            assert self.args.rank == -1

        random.seed(self.args.manual_seed)
        np.random.seed(self.args.manual_seed)
        torch.manual_seed(self.args.manual_seed)
        if self.args.n_gpu > 0:
            torch.cuda.manual_seed_all(self.args.manual_seed)

        if use_cuda:
            if torch.cuda.is_available():
                if self.args.local_rank == -1:
                    self.device = torch.device('cuda')
                else:
                    self.device = torch.device('cuda', self.args.local_rank)
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    "Make sure CUDA is available or set `use_cuda=False`."
                )
        else:
            self.device = "cpu"
        print("setting device complete. device:", self.device)

        if not use_cuda:
            self.args.fp16 = False
        
        ### load model and tokenizer
        _config_class, _model_class, _tokenizer_class, no_decay = MODEL_CLASSES[model_type]
        self.no_decay = no_decay
        if self.args.disable_dropout:
            self.language_model = _model_class.from_pretrained(model_name, attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0, summary_first_dropout=0.0)
        else:
            self.language_model = _model_class.from_pretrained(model_name)
        self.lm_tokenizer = _tokenizer_class.from_pretrained(model_name)
        
        if self.args.fresh_tokenizer:
            # vocab_file should be a dictionary mapping tokens to ids, and data files should be pre-tokenized into ids.
            self.args.pad_token_id = vocab_file["<pad>"]
            vocab_size_ = len(vocab_file)
            pad_token_id = self.args.pad_token_id
        else:
            # use default tokenizer and add tokens into vocab if necessary
            self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token
            self.lm_tokenizer.pad_token_id = self.lm_tokenizer.eos_token_id
            if vocab_file:
                # add new tokens
                self.lm_tokenizer.add_tokens(vocab_file)
            vocab_size_ = len(self.lm_tokenizer)
            pad_token_id = self.lm_tokenizer.pad_token_id
        print("\tpad token id:", pad_token_id)        

        if init_weights:
            init_param = dict()
            temp_config = self.language_model.config
            assert type(self.args.n_layer) == int
            temp_config.n_layer = self.args.n_layer
            init_param['n_layer'] = self.args.n_layer
            # some optional ones
            if self.args.n_embd:
                temp_config.n_embd = self.args.n_embd
                init_param['n_embd'] = self.args.n_embd
            if self.args.n_head:
                temp_config.n_head = self.args.n_head
                init_param['n_head'] = self.args.n_head
            if self.args.untie_word_embeddings:
                temp_config.tie_word_embeddings = False
                init_param['tie_word_embeddings'] = False
            temp_config.vocab_size = vocab_size_       
            init_param['vocab_size'] = temp_config.vocab_size

            if self.args.memory_block:
                assert self.args.wernicke_broca and self.args.num_wernicke_layer == 1
                temp_config.memory_block = True
                init_param['memory_block'] = True
                if self.args.memory_block_top_k:
                    assert self.args.memory_block_top_k > 0
                    temp_config.memory_block_top_k = self.args.memory_block_top_k
                    init_param['memory_block_top_k'] = temp_config.memory_block_top_k

            if self.args.first_mlp_width:
                assert type(self.args.first_mlp_width) == int
                temp_config.first_mlp_width = self.args.first_mlp_width
                init_param['first_mlp_width'] = temp_config.first_mlp_width

            self.language_model = _model_class(temp_config)
            print("***initing weights*** params:", init_param)

        # resize the embeddings, update config since new tokens are perhaps added
        self.language_model.resize_token_embeddings(vocab_size_)   # TODO: , pad_to_multiple_of=8
        embedding_shape = self.language_model.get_input_embeddings().weight.shape
        self.language_model.config.vocab_size = embedding_shape[0]
        print("***embedding shape***:", embedding_shape)

        if self.args.init_wernicke_mlp_keys:
            assert self.args.memory_block
            with open("{}/{}train.json".format(self.args.data_dir_parent, self.args.data_dir)) as f:
                train_da = json.load(f)
            all_entities = set()
            for item in tqdm(train_da):
                all_entities.add(tuple(item['input_text'][0]))
                all_entities.add(tuple(item['target_text'][0]))
            all_entities = torch.tensor(list(all_entities))
            
            entity_keys = torch.mean(self.language_model.transformer.wte(all_entities), dim=-2)
            other_keys = self.language_model.transformer.wte.weight.data[-14:, :]
            all_keys = torch.cat((entity_keys, other_keys), dim=0)
            self.language_model.transformer.h[0].softmax_mlp.c_fc.weight.data[:all_keys.shape[0], :] = all_keys.data

        self.entity2id = None
        if self.args.eval_full_negatives and self.args.num_wernicke_layer > 0:
            with open("{}/{}train.json".format(self.args.data_dir_parent, self.args.data_dir)) as f:
                train_da = json.load(f)
            all_entities = set()
            for item in tqdm(train_da):
                all_entities.add(tuple(item['input_text'][0] + [pad_token_id]))
                all_entities.add(tuple(item['target_text'][0] + [pad_token_id]))
            entity2id = dict()
            concept_positions = []
            for (e1, e2, _) in all_entities:
                entity2id[(e1, e2)] = len(entity2id)
                concept_positions.append([0,1,0])
            self.all_entities = torch.tensor(list(all_entities))
            self.concept_positions = torch.tensor(concept_positions)
            self.entity2id = entity2id

        if self.args.wernicke_broca:
            print("***adding wernicke and broca modules***")
            wb_params = {
                "num_wernicke_layer": self.args.num_wernicke_layer,
                "num_broca_layer": self.args.num_broca_layer,
                "max_patch_length": self.args.max_patch_length,
                "concept_ln_type": self.args.concept_ln_type,
            }
            print(wb_params)
            self.language_model.add_wernicke_broca(wb_params)

            if os.path.isdir(model_name):
                files = os.listdir(model_name)
                if "concept_ln.pth" in files:
                    assert "broca_query.pth" in files
                    print("***loading pretrained concept_ln and broca_query***")
                    self.language_model.concept_ln = torch.load(os.path.join(model_name, "concept_ln.pth"))
                    self.language_model.broca_query = torch.load(os.path.join(model_name, "broca_query.pth"))
                    if "memory_block.pth" in files:
                        print("***loading pretrained memory_block***")
                        self.language_model.transformer.h[0] = torch.load(os.path.join(model_name, "memory_block.pth"))

        self.model = Seq2SeqModule(
            language_model=self.language_model,
        )

        self.args.model_type = model_type
        self.args.model_name = model_name

        print("### general model args:")
        print(self.args)
        print("### lm config:")
        print(self.language_model.config)


    def train_model(
        self,
        train_data,
        output_dir=None,
        show_running_loss=True,
        args=None,
        eval_data=None,
        test_data=None,
        verbose=True,
        **kwargs,
    ):
        if args:
            self.args.update_from_dict(args)

        if self.distributed:
            self.args.silent = (self.args.rank != 0)

        if self.args.silent:
            show_running_loss = False

        if self.args.evaluate_during_training and eval_data is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_data is not specified."
                " Pass eval_data to model.train_model() if using evaluate_during_training."
            )

        assert output_dir
        if os.path.exists(output_dir) and os.listdir(output_dir) and not self.args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set args.overwrite_output_dir = True to overcome.".format(output_dir)
            )

        self._move_model_to_device()

        train_dataset = self.load_and_cache_examples(train_data)

        os.makedirs(output_dir, exist_ok=True)

        global_step, training_details = self.train(
            train_dataset,
            output_dir,
            show_running_loss=show_running_loss,
            eval_data=eval_data,
            test_data=test_data,
            verbose=verbose,
            **kwargs,
        )
        
        if verbose:
            logger.info(" Training of {} model complete. Saved to {}.".format(self.args.model_name, output_dir))

        return global_step, training_details

    def train(
        self,
        train_dataset,
        output_dir,
        show_running_loss=True,
        eval_data=None,
        test_data=None,
        verbose=True,
        **kwargs,
    ):
        model = self.model
        args = self.args
        lm_tokenizer = self.lm_tokenizer

        generator = torch.Generator()
        generator.manual_seed(args.manual_seed)

        if self.distributed:
            print("invoking distributed sampler for rank", args.rank)
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
        else:
            train_sampler = RandomSampler(train_dataset, generator=generator)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
            drop_last=True,
        )

        if args.evaluate_during_training:
            eval_dataloader_dict = dict()
            for split_name in eval_data:
                eval_dataset = self.load_and_cache_examples(eval_data[split_name])
                if self.distributed:
                    eval_sampler = DistributedSampler(eval_dataset, shuffle=True)
                else:
                    eval_sampler = SequentialSampler(eval_dataset)

                eval_dataloader = DataLoader(
                    eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
                )
                eval_dataloader_dict[split_name] = eval_dataloader

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = (
                args.max_steps
                // (len(train_dataloader) // args.gradient_accumulation_steps)
                + 1
            )
        else:
            t_total = (
                len(train_dataloader)
                // args.gradient_accumulation_steps
                * args.num_train_epochs
            )

        no_decay = self.no_decay

        freeze_keyword = []
        if self.args.memory_block_value_only:
            assert self.args.memory_block
            freeze_keyword.extend(['wte', 'wpe', 'softmax_mlp.c_fc'])

        optimizer_grouped_parameters = []
        optimizer_grouped_parameters.extend(
            [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                        and not any(nd in n for nd in freeze_keyword)
                    ],
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)
                        and not any(nd in n for nd in freeze_keyword)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        )

        num_total_params = 0
        for n, p in model.named_parameters():
            print(n, p.shape, p.requires_grad)
            num_total_params += p.numel()
        print("number of total params:", num_total_params)
        print("==========================")

        params_to_clip = []
        num_total_params_optimized = 0
        for pg in optimizer_grouped_parameters:
            params_to_clip.extend(pg['params'])
            for p in pg['params']:
                num_total_params_optimized += p.numel()
        print("number of optimized params:", num_total_params_optimized)

        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        args.warmup_steps = (
            warmup_steps if args.warmup_steps == 0 else args.warmup_steps
        )

        print("****************begin training. Total # of steps:", t_total, "warmup steps:", args.warmup_steps, "epochs:", args.num_train_epochs)
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon,
            betas=args.adam_betas,
        )
        
        if args.scheduler == "constant_schedule_with_warmup":
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps
            )
        elif args.scheduler == "linear_schedule_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
            )
        elif args.scheduler == "cosine_schedule_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )
        else:
            raise ValueError("{} is not a valid scheduler.".format(args.scheduler))

        if self.distributed:
            # DDP
            if args.local_rank == -1:
                temp = 0
            else:
                temp = args.local_rank
            model = DDP(model, device_ids=[temp], output_device=temp)

        # in the distributed case, disable prints for non-master nodes
        if self.distributed:
            if args.rank != 0:
                print("I'm rank {}. I'm muted from now on.".format(args.rank))
                def print_pass(*args_):
                    pass
                builtins.print = print_pass
            else:
                print("I'm rank {}. I'll continue to print.".format(args.rank))


        logger.info(" Training started")

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0

        model.zero_grad()
        optimizer.zero_grad()
        
        train_iterator = trange(
            int(args.num_train_epochs), desc="Epoch", disable=args.silent, mininterval=0
        )
        epoch_number = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0

        if args.model_name and os.path.exists(args.model_name):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name.split("/")[-1].split("-")
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (
                    len(train_dataloader) // args.gradient_accumulation_steps
                )
                steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // args.gradient_accumulation_steps
                )

                logger.info(
                    "   Continuing training from checkpoint, will skip to saved global_step"
                )
                logger.info("   Continuing training from epoch %d", epochs_trained)
                logger.info("   Continuing training from global step %d", global_step)
                logger.info(
                    "   Will skip the first %d steps in the current epoch",
                    steps_trained_in_current_epoch,
                )
            except ValueError:
                logger.info("   Starting fine-tuning.")

        if args.evaluate_during_training:
            training_progress_scores = self._create_training_progress_scores(**kwargs)

        if args.fp16:
            from torch.cuda import amp
            scaler = amp.GradScaler()

        for current_epoch in train_iterator:

            current_epoch_losses = torch.zeros(1).to(self.device)
            steps_avg = 0

            model.train()

            if self.distributed:
                train_dataloader.sampler.set_epoch(current_epoch)

            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            train_iterator.set_description(
                f"Epoch {epoch_number + 1} of {args.num_train_epochs}"
            )
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Running Epoch {epoch_number} of {args.num_train_epochs}",
                disable=args.silent,
                mininterval=0,
            )

            for step, batch in enumerate(batch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                inputs = self._get_inputs_dict(batch)
                # for key in inputs:
                #     print(key, inputs[key].shape, inputs[key][:2])
                # print("===")

                if args.fp16:
                    with amp.autocast():
                        outputs = model(**inputs, ban_pos_encoding=BAN_POS_ENCODING)
                else:
                    outputs = model(**inputs, ban_pos_encoding=BAN_POS_ENCODING)

                if args.wernicke_broca:
                    if args.wb_lmloss:
                        loss = outputs.loss
                    else:
                        loss = outputs.concept_loss
                else:
                    loss = outputs.loss
                current_epoch_losses[0] += loss.item()
                steps_avg += 1

                if show_running_loss:
                    if args.wernicke_broca:
                        batch_iterator.set_description(
                            f"Epochs {epoch_number+1}/{args.num_train_epochs}. Concept: {outputs.concept_loss.item():9.4f}|Loss: {outputs.loss.item():9.4f}"  
                        )
                    else:
                        batch_iterator.set_description(
                            f"Epochs {epoch_number+1}/{args.num_train_epochs}. Loss: {outputs.loss.item():9.4f}"  
                        )
            
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    
                    if args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    optimizer.zero_grad()

                    global_step += 1
                    if ((args.save_steps > 0) and (global_step % args.save_steps == 0)) or (args.save_step_dense>0 and global_step % args.save_step_dense_interval == 0 and global_step<=args.save_step_dense):
                        output_dir_current = os.path.join(
                            output_dir, "checkpoint-{}".format(global_step)
                        )

                        self.save_model(
                            output_dir_current, optimizer, scheduler, model=model
                        )
                        
                        if args.evaluate_during_training:
                            results = self.eval_model(
                                eval_dataloader_dict,
                                verbose=verbose,
                                silent=args.evaluate_during_training_silent,
                                **kwargs,
                            )
                            training_progress_scores["global_step"].append(global_step)
                            training_progress_scores["epoch"].append(-1)
                            for key in results:
                                if key not in training_progress_scores:
                                    training_progress_scores[key] = []
                                training_progress_scores[key].append(results[key])
                            report = pd.DataFrame(training_progress_scores)
                            report.to_csv(
                                os.path.join(output_dir, "training_progress_scores.csv"),
                                index=False,
                            )

                            if (not self.distributed) and args.predict_during_training:
                                self.predict(test_data, output_dir_current, skip_model_moving=True)

                            model.train()
                    
            current_epoch_losses[0] /= steps_avg
            if self.distributed:
                dist.all_reduce(current_epoch_losses, op=dist.ReduceOp.AVG)

            print("current_epoch_running_losses", current_epoch_losses)
            
            epoch_number += 1
            output_dir_current = os.path.join(
                output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number)
            )

            if args.save_model_every_epoch:
                os.makedirs(output_dir_current, exist_ok=True)
                self.save_model(output_dir_current, optimizer, scheduler, model=model)

                if args.evaluate_during_training:
                    results = self.eval_model(
                        eval_dataloader_dict,
                        verbose=verbose,
                        silent=args.evaluate_during_training_silent,
                        **kwargs,
                    )

                    print(results)

                    training_progress_scores["global_step"].append(global_step)
                    training_progress_scores["epoch"].append(epoch_number)
                    for key in results:
                        if key not in training_progress_scores:
                            training_progress_scores[key] = []
                        training_progress_scores[key].append(results[key])
                    report = pd.DataFrame(training_progress_scores)
                    report.to_csv(
                        os.path.join(output_dir, "training_progress_scores.csv"),
                        index=False,
                    )

                    if (not self.distributed) and args.predict_during_training:
                        self.predict(test_data, output_dir_current, skip_model_moving=True)
         
        return (
            global_step,
            tr_loss / global_step
            if not self.args.evaluate_during_training
            else training_progress_scores,
        )

    def eval_model(
        self, eval_dataloader_dict, verbose=True, silent=False, **kwargs
    ):
        model = self.model
        args = self.args
        model.eval()

        results = {}

        if args.eval_full_negatives:
            if args.num_wernicke_layer == 0:
                # last token is pad
                concept_embeds = model.lm.transformer.wte.weight[:-1, :]
            else:
                bsz = 512
                concept_embeds = []
                with torch.no_grad():
                    for i in range(0, self.all_entities.shape[0], bsz):
                        concept_embeds.append(model(input_ids=self.all_entities[i:i + bsz, :].to(self.device), concept_positions=self.concept_positions[i:i + bsz, :].to(self.device), ban_pos_encoding=BAN_POS_ENCODING, return_act_wernicke_layer=True).squeeze())
                    concept_embeds = torch.cat(concept_embeds, dim=0)

        for split_name in eval_dataloader_dict:
            eval_dataloader = eval_dataloader_dict[split_name]

            LM_loss = torch.zeros(4).to(self.device)
            LM_loss_full = torch.zeros(4).to(self.device)
            total_samples = 0
            
            if args.fp16:
                from torch.cuda import amp

            for batch in tqdm(
                eval_dataloader, disable=args.silent or silent, desc="Running Evaluation"
            ):
                inputs = self._get_inputs_dict(batch)
                with torch.no_grad():
                    if args.fp16:
                        with amp.autocast():
                            outputs = model(**inputs, ban_pos_encoding=BAN_POS_ENCODING)
                    else:
                        outputs = model(**inputs, ban_pos_encoding=BAN_POS_ENCODING)
                    if args.wernicke_broca:
                        LM_loss[0] += outputs.loss.item() * len(batch)
                        LM_loss[1] += outputs.mrr.item() * len(batch)
                        LM_loss[2] += outputs.concept_loss.item() * len(batch)
                        LM_loss[3] += outputs.concept_mrr.item() * len(batch)
                        
                        # eval on the full negative
                        if args.eval_full_negatives:
                            if args.fp16:
                                with amp.autocast():
                                    outputs = model(**inputs, ban_pos_encoding=BAN_POS_ENCODING, in_batch_negative=False, concept_embeds=concept_embeds)   
                            else:
                                outputs = model(**inputs, ban_pos_encoding=BAN_POS_ENCODING, in_batch_negative=False, concept_embeds=concept_embeds)
                            LM_loss_full[2] += outputs.concept_loss.item() * len(batch)
                            LM_loss_full[3] += outputs.concept_mrr.item() * len(batch)
                        # --

                    else:
                        LM_loss[0] += outputs.loss.item() * len(batch)
                        LM_loss[1] += outputs.mrr.item() * len(batch)
                total_samples += len(batch)

            LM_loss = LM_loss/total_samples
            LM_loss_full = LM_loss_full/total_samples
            if self.distributed:
                dist.all_reduce(LM_loss, op=dist.ReduceOp.AVG)

            results["{}.{}".format(split_name, "loss")] = LM_loss[0].cpu().item()
            results["{}.{}".format(split_name, "mrr")] = LM_loss[1].cpu().item()
            results["{}.{}".format(split_name, "concept_loss")] = LM_loss[2].cpu().item()
            results["{}.{}".format(split_name, "concept_mrr")] = LM_loss[3].cpu().item()

            # --
            if args.eval_full_negatives:
                assert args.wernicke_broca
                results["{}.{}".format(split_name, "concept_loss_full")] = LM_loss_full[2].cpu().item()
                results["{}.{}".format(split_name, "concept_mrr_full")] = LM_loss_full[3].cpu().item()

        return results

    def load_and_cache_examples(self, data):
        return SimpleDataset(self.lm_tokenizer, self.args, data, entity2id=self.entity2id)

    def _create_training_progress_scores(self, **kwargs):
        extra_metrics = {key: [] for key in kwargs}
        training_progress_scores = {
            "global_step": [],
            "epoch": [],
            **extra_metrics,
        }

        return training_progress_scores

    def save_model(
        self,
        output_dir=None,
        optimizer=None,
        scheduler=None,
        model=None,
    ):
        
        if self.distributed and self.args.rank != 0:
            # no saving for non-master nodes
            return

        assert output_dir
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Saving model into {output_dir}")

        if model and not self.args.dont_save_models:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            
            # self.save_model_args(output_dir)
            os.makedirs(os.path.join(output_dir), exist_ok=True)
            model_to_save.save_pretrained(output_dir)
            self.lm_tokenizer.save_pretrained(os.path.join(output_dir))

            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            if optimizer and scheduler and self.args.save_optimizer_and_scheduler:
                torch.save(
                    optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                )
                torch.save(
                    scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                )

    def _move_model_to_device(self):
        self.model.to(self.device)

    def _get_inputs_dict(self, batch):
        return {key: batch[key].to(self.device) for key in batch}

    # def save_model_args(self, output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    #     self.args.save(output_dir)


def main(
    model_type: str,
    model_name: str,
    output_dir: str,
    init_weights: bool=False,
    add_tokens: bool=False,
    **model_args,
):  
    train_df = read_json(os.path.join(model_args['data_dir'], "train.json"))
    eval_df = read_json(os.path.join(model_args['data_dir'], "valid.json"))
    test_df = None
    if 'predict_during_training' in model_args and model_args['predict_during_training']:
        test_df = read_json(os.path.join(model_args['data_dir'], "test.json"))

    vocab_file=None
    if add_tokens:
        with open(os.path.join(model_args['data_dir'], "vocab.json")) as f:
            vocab_file = json.load(f)

    m = Seq2SeqModel(
        model_type=model_type,
        model_name=model_name,
        args=model_args,
        init_weights=init_weights,
        vocab_file=vocab_file,
        use_cuda=True,
    )
    m.train_model(train_data=train_df, eval_data=eval_df, test_data=test_df, output_dir=output_dir)


if __name__ == "__main__":
    fire.Fire(main)