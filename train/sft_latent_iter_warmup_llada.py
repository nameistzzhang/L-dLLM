import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
import math
import shutil
import time
from pathlib import Path
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm


from train.utils import get_config, flatten_omega_conf, AverageMeter
from models import LLaDAModelLM
from train.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error



logger = get_logger(__name__, log_level="INFO")



class TrainDataset(Dataset):
    def __init__(self, inputs, labels, pmasks):
        self.inputs = inputs
        self.labels = labels
        self.pmasks = pmasks

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            self.inputs[idx],
            self.labels[idx],
            self.pmasks[idx]
        )



def main():

    # * ---- load and parse configuration ----
    config = get_config()
    

    # * ---- set up training environment ----
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.project) / "logs") # set logging dir to project/logs for accelerator tracking    

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=False, # * split_batches=True ensures the dataloader yields batches that will be split across devices. Here we want standard DDP behavior.
    ) # set up accelerator for distributed training and mixed precision
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    ) # set up logging format and level

    logger.info(accelerator.state, main_process_only=False) # log accelerator state for debugging

    if accelerator.is_local_main_process: # only the main process will log info
        set_verbosity_info()
    else:
        set_verbosity_error()

    if accelerator.is_main_process: # save configuration into project folder for reproducibility
        os.makedirs(config.experiment.project, exist_ok=True)
        config_path = Path(config.experiment.project) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    if config.training.seed is not None: # set up random seed for reproducibility
        set_seed(config.training.seed)


    # * ---- configure logging and tracking with wandb ----
    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id
        run_name = config.wandb.get("run_name", None)

        wandb_init_kwargs = dict(
            name=run_name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint", None)

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )


    # * ---- load model and tokenizer ----
    logger.info("Loading models and optimizer")

    pretrained_model = config.model.pretrained_model
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model) # load tokenizer
    uni_prompting = UniversalPrompting(tokenizer, max_prompt_len=config.training.max_prompt_len,
                                       max_gen_length=config.training.max_gen_length,
                                       ignore_id=-100) # set up universal prompting for data processing
    
    model = LLaDAModelLM.from_pretrained(pretrained_model, torch_dtype=torch.bfloat16)
    model = model.to(accelerator.device) # load model to accelerator device (GPU/TPU)

    mask_id = tokenizer.encode('<|mdm_mask|>')[0]
    pad_id = tokenizer.encode('<|endoftext|>')[0]


    # * ---- set up optimizer ----
    optimizer_config = config.optimizer.params

    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ] # set up parameter groups for optimizer with weight decay applied to all parameters except those in no_decay list

    optimizer_type = config.optimizer.name

    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        ) # set up AdamW optimizer with parameters from config
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")


    # * ---- util function for data processing ----
    @torch.no_grad()
    def prepare_inputs_and_labels_for_text(
        prompt, response, step_map, eps=1e-3, mask_id=mask_id
    ):
        input_ids_lm, labels_lm, start_pos, drop_num = uni_prompting((prompt, response))
        
        B, L = input_ids_lm.shape
        max_gen_len = config.training.max_gen_length
        if max_gen_len + start_pos < L:
            L_after = start_pos + max_gen_len
        else:
            L_after = L
        input_ids_lm = input_ids_lm[:, :L_after]
        labels_lm = labels_lm[:, :L_after]

        lower = config.training.lower_p
        upper = config.training.upper_p    

        m = config.training.mask_times_per_sample
        B, L = input_ids_lm.shape
        device = input_ids_lm.device

        noisy_list, label_list, pmask_list = [], [], []
        for b in range(B):
            base_ids  = input_ids_lm[b]
            label_ids = labels_lm[b]

            if config.training.post_num is not None:
                pad_mask_b = (base_ids == pad_id)
                pad_mask_b[:start_pos] = False
                keep_first_pad_b = pad_mask_b & (torch.cumsum(pad_mask_b.int(), dim=0) <= config.training.post_num)
                tail_pad_b       = pad_mask_b & ~keep_first_pad_b
            else:
                keep_first_pad_b = torch.zeros(L, dtype=torch.bool, device=device)
                tail_pad_b       = torch.zeros(L, dtype=torch.bool, device=device)

            for _ in range(m):
                t = (upper - lower) * torch.rand(1, device=device) + lower
                rand_mask = torch.rand(L, device=device) < t
                rand_mask[:start_pos] = False
                rand_mask = rand_mask & ~tail_pad_b

                if not rand_mask.any():
                    continue

                noisy_ids = base_ids.clone()
                noisy_ids[rand_mask]   = mask_id
                noisy_ids[tail_pad_b]  = mask_id

                noisy_list.append(noisy_ids)
                label_list.append(label_ids)
                pmask_list.append(rand_mask)

        noisy_batch = torch.stack(noisy_list)    # (B*m, L)
        labels_lm   = torch.stack(label_list)
        p_mask      = torch.stack(pmask_list)
        
        valid_rows = p_mask.any(dim=1)
        noisy_batch = noisy_batch[valid_rows]
        labels_lm   = labels_lm[valid_rows]
        p_mask      = p_mask[valid_rows]
        
        return noisy_batch, labels_lm, p_mask, start_pos, drop_num
    

    # * ---- util function for collating batches ----
    def simple_collate(batch):
        inp, lbl, msk = zip(*batch)  
        return {
            "input_ids":  torch.stack(inp),
            "labels":     torch.stack(lbl),
            "p_mask_lm":  torch.stack(msk)
        }


    # * ---- set up dataloader ----
    logger.info("Creating dataloaders and lr_scheduler")

    with open("./data/" + config.dataset.optimization_data + ".json", 'r') as f:
        dataset_load = json.load(f) # load dataset from json file

    prompt_list = []
    response_list = []
    step_map_list = []
    for x in dataset_load:
        prompt_list.append(x["prompt"])
        response_list.append(x["response"])
        if "step_map" not in x.keys():
            step_map_list.append([j for j in range(config.training.max_gen_length)])
        else:
            step_map_list.append(x["step_map"])
    input_ids, labels, p_mask_lm, start_pos, drop_num = prepare_inputs_and_labels_for_text(prompt_list, response_list, step_map_list)
    dataset_lm = TrainDataset(input_ids, labels, p_mask_lm)

    train_dataloader_lm = DataLoader(
        dataset_lm,
        batch_size=config.training.batch_size_lm,
        sampler=None,
        collate_fn=simple_collate,
        num_workers=0
    )


    # * ---- set up learning rate scheduler ----
    total_batch_size_lm = config.training.batch_size_lm * accelerator.num_processes * config.training.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(dataset_lm) / total_batch_size_lm)
    num_train_epochs = config.training.num_train_epochs
    max_train_steps = num_update_steps_per_epoch * num_train_epochs + 1

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale
    )


    # * ---- prepare model, optimizer, lr_scheduler and dataloader with accelerator ----
    logger.info("Preparing model, optimizer and dataloaders")
    #model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    model, optimizer, lr_scheduler, train_dataloader_lm = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader_lm
    )
    
    # * ---- Resume from checkpoint ----
    first_epoch = 0
    if config.experiment.get("resume_from_checkpoint"):
        if config.experiment.resume_from_checkpoint != "latest":
            path = config.experiment.resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = [d for d in Path(config.experiment.project).iterdir() if d.is_dir() and d.name.startswith("checkpoint")]
            dirs.sort(key=lambda x: os.path.getmtime(x))
            path = dirs[-1] if dirs else None

        if path is not None:
            accelerator.print(f"Resuming from checkpoint: {path}")
            accelerator.load_state(path)
            # Parse epoch from checkpoint name if possible
            if "checkpoint-epoch-" in str(path):
                try:
                    first_epoch = int(str(path).split("checkpoint-epoch-")[-1])
                    logger.info(f"Resuming from epoch {first_epoch}")
                except ValueError:
                    logger.warning(f"Could not parse epoch from checkpoint path: {path}")



    # * ---- training info logging ----
    logger.info("***** Running training *****")
    logger.info(f"  Num response = {len(dataset_load)}")
    logger.info(f"  Num sample dropped = {drop_num}")
    logger.info(f"  Num training data = {input_ids.shape[0]}")
    logger.info(f"  Num training steps = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.training.batch_size_lm}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size_lm}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    
    
    # * ---- training loss function ----
    """
    Calculates the Masked Negative Log-Likelihood (NLL) Loss.
    1. Computes log-probabilities for the input tokens.
    2. Gathers the log-probabilities corresponding to the target labels.
    3. Masks out positions that should be ignored using p_mask_lm.
    4. Computes the average negative log-likelihood per valid token for each sequence.
    5. Returns the mean loss across the batch.
    """
    def forward_process(input_ids, labels, p_mask_lm):
        logits = model(input_ids).logits
        B, T, V = logits.shape

        safe_labels = labels.clone()
        safe_labels[labels == -100] = 0

        log_probs = F.log_softmax(logits, dim=-1)   # (B, T, V)
        logp_tok = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)     # (B, T)
        loss_lm = - (logp_tok * p_mask_lm).sum(dim=1)

        mask_num = (p_mask_lm).sum(dim=1).clamp(min=1)
        loss_lm = loss_lm / mask_num
    
        loss_lm = loss_lm.sum() / B
        return loss_lm


    # * ---- training loop ----
    for epoch in range(first_epoch, num_train_epochs):
        
        model.train()
        
        progress_bar = tqdm(
            train_dataloader_lm,
            desc=f"Epoch {epoch+1}/{num_train_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,    
            leave=True          
        )
        
        for step, batch in enumerate(progress_bar, start=1):
            global_step = epoch * len(train_dataloader_lm) + step
            
            input_ids = batch["input_ids"].to(accelerator.device)
            labels    = batch["labels"].to(accelerator.device)
            p_mask_lm = batch["p_mask_lm"].to(accelerator.device)

            # Accumulate gradients manually
            loss_lm = forward_process(
                input_ids=input_ids,
                labels=labels,
                p_mask_lm=p_mask_lm
            )
            
            # Record unscaled loss for logging
            step_loss = loss_lm.detach().float()

            loss_lm = loss_lm / accelerator.gradient_accumulation_steps
            accelerator.backward(loss_lm)

            if global_step % accelerator.gradient_accumulation_steps == 0: # update model parameters and log training info
                if config.training.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(),
                                                config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.is_local_main_process:
                   print(loss_lm)

                accelerator.log({
                    "loss": step_loss.item(),
                    "lr": lr_scheduler.get_last_lr()[0]
                }, step=global_step)
            
            del input_ids, labels, p_mask_lm # release memory
            torch.cuda.empty_cache()

        # Save checkpoint at the end of each epoch
        output_dir = Path(config.experiment.project) / f"checkpoint-epoch-{epoch+1}"
        accelerator.save_state(output_dir)
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
                safe_serialization=True
            )
            tokenizer.save_pretrained(output_dir)
        logger.info(f"Epoch {epoch+1} checkpoint saved to {output_dir}")

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
