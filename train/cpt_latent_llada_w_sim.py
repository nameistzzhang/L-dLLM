import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
import math
import datetime
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
from models import LatentLLaDAModelLM
from train.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error



logger = get_logger(__name__, log_level="INFO")



class TrainDataset(Dataset):
    def __init__(self, inputs, labels, pmasks, config):
        self.inputs = inputs
        self.labels = labels
        self.pmasks = pmasks
        self.config = config

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            self.inputs[idx],
            self.labels[idx],
            self.pmasks[idx]
        )

    @staticmethod
    def simple_collate(batch):
        inp, lbl, msk = zip(*batch)  
        return {
            "input_ids":  torch.stack(inp),
            "labels":     torch.stack(lbl),
            "p_mask_lm":  torch.stack(msk)
        }



def main():

    # * ---- load and parse configuration ----
    config = get_config()
    

    # * ---- set up training environment ----
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # Initialize wandb run name if not present
    if config.wandb.get("run_name") is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config.wandb.run_name = f"run_{timestamp}"
    
    # Create output directory based on project and run name
    config.experiment.output_dir = str(Path(config.experiment.project) / config.wandb.run_name)
    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")

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
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
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
                                       ignore_id=-100, post_num=config.training.post_num) # set up universal prompting for data processing
    
    model = LatentLLaDAModelLM.from_pretrained(pretrained_model, torch_dtype=torch.bfloat16)
    model = model.to(accelerator.device) # load model to accelerator device (GPU/TPU)

    mask_id = tokenizer.encode('<|mdm_mask|>')[0]
    pad_id = tokenizer.encode('<|endoftext|>')[0]


    # * ---- set up optimizer ----
    optimizer_config = config.optimizer.params

    if config.training.get("warmup", False):
        logger.info("Warmup mode enabled: only training the latent gate parameters.")
        
        trainable_param_names = []
        for name, param in model.named_parameters():
            if "model.gate." in name:
                param.requires_grad = True
                trainable_param_names.append(name)
            else:
                param.requires_grad = False
                
        if accelerator.is_local_main_process:
            logger.info("=== Trainable parameters during warmup ===")
            for p_name in trainable_param_names:
                logger.info(f"  - {p_name}")
            logger.info("==========================================")

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
    def prepare_all_noisy_batch(
        prompt, response, step_map, eps=1e-3, mask_id=mask_id
    ):
        """
        Prepares a batch of noisy data where the entire response is masked.
        
        Args:
            prompt (List[str]): List of prompt strings.
            response (List[str]): List of response strings.
            step_map (List[List[int]]): List of step maps (currently unused in body but required by signature).
            eps (float, optional): Epsilon value. Defaults to 1e-3.
            mask_id (int, optional): The token ID used for masking. Defaults to mask_id from outer scope.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]: 
                - noisy_batch: Input IDs with responses fully masked. Shape: (B*m, L)
                - labels_lm: Original labels for loss calculation. Shape: (B*m, L)
                - p_mask: Boolean mask indicating prediction positions (masked tokens). Shape: (B*m, L)
                - start_pos: The starting position of the response.
                - drop_num: Number of samples dropped during processing.
        """

        input_ids_lm, labels_lm, start_pos, drop_num = uni_prompting((prompt, response))
        
        B, L = input_ids_lm.shape
        max_gen_len = config.training.max_gen_length
        if max_gen_len + start_pos < L:
            L_after = start_pos + max_gen_len
        else:
            L_after = L
        input_ids_lm = input_ids_lm[:, :L_after]
        labels_lm = labels_lm[:, :L_after]

        m = 1 # for each sample, only 1 fully masked version is created. 
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
                rand_mask = torch.ones(L, dtype=torch.bool, device=device) # * mask all response
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
    input_ids, labels, p_mask_lm, start_pos, drop_num= prepare_all_noisy_batch(prompt_list, response_list, step_map_list)
    dataset_lm = TrainDataset(input_ids, labels, p_mask_lm, config) # create dataset for language modeling with noisy inputs and corresponding labels and masks

    train_dataloader_lm = DataLoader(
        dataset_lm,
        batch_size=config.training.batch_size_lm,
        shuffle=True,
        collate_fn=TrainDataset.simple_collate,
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
        power=config.lr_scheduler.params.power,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale
    )


    # * ---- prepare model, optimizer, lr_scheduler and dataloader with accelerator ----
    logger.info("Preparing model, optimizer and dataloaders")
    #model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    model, optimizer, train_dataloader_lm = accelerator.prepare(
        model, optimizer, train_dataloader_lm
    )
    
    # * ---- Resume from checkpoint ----
    first_epoch = 0
    global_update_step = 0
    if config.experiment.get("resume_from_checkpoint"):
        if config.experiment.resume_from_checkpoint != "latest":
            path = config.experiment.resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = [d for d in Path(config.experiment.output_dir).iterdir() if d.is_dir() and d.name.startswith("checkpoint")]
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
                    
                    # Calculate global_update_step based on resumed epoch
                    steps_per_epoch = len(train_dataloader_lm) // config.training.gradient_accumulation_steps
                    global_update_step = first_epoch * steps_per_epoch
                    logger.info(f"Resuming global step from {global_update_step}")
                    
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
    
    
    # * ---- training forward logit and loss function ----
    """
    Calculates the Masked Negative Log-Likelihood (NLL) Loss.
    1. Computes log-probabilities for the input tokens.
    2. Gathers the log-probabilities corresponding to the target labels.
    3. Masks out positions that should be ignored using p_mask_lm.
    4. Computes the average negative log-likelihood per valid token for each sequence.
    5. Returns the mean loss across the batch.
    """
    def unnmask_forward_process(input_ids, labels, p_mask_lm):
        
        logits = model(input_ids).logits
        B, T, V = logits.shape

        safe_labels = labels.clone()
        safe_labels[labels == -100] = 0

        # Calculate accuracy
        preds = torch.argmax(logits, dim=-1) # (B, T)
        correct_mask = (preds == safe_labels) & p_mask_lm
        accuracy = correct_mask.sum().float() / p_mask_lm.sum().clamp(min=1).float()

        raw_loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100, reduction='none')
        loss_lm = (raw_loss * p_mask_lm).sum(dim=1)

        mask_num = p_mask_lm.sum(dim=1).clamp(min=1)
        loss_lm = (loss_lm / mask_num).sum() / B
        
        if accelerator.is_local_main_process:
            logger.info(f"unnmask loss: {loss_lm.item()}")
            logger.info(f"unnmask accuracy: {accuracy.item()}")
            
        return loss_lm, accuracy, logits
    
    def flowmatch_forward_process(input_ids, probs, t, labels, weighting_strategy="constant"):
        """
        Computes the Flow Matching loss.

        Args:
            input_ids (torch.LongTensor): (Batch, Seq).
            probability (torch.Tensor): (Batch, Seq, Vocab).
            t (torch.Tensor): (Batch, 1).
            labels (torch.LongTensor): (Batch, Seq). Indices set to -100 are ignored.
        
        Returns:
            loss_lm (torch.Tensor): Scalar loss.
            accuracy (float): Scalar accuracy.
        """

        # Construct alpha_mask: 0 for tokens where label is -100, 1 otherwise
        target_dtype = next(model.parameters()).dtype
        alpha_mask = (labels != -100).to(target_dtype).unsqueeze(-1) # (Batch, Seq, 1)
        t_for_model = t.to(target_dtype)

        logits = model(
            input_ids=input_ids,
            labels=labels,
            probability=probs,
            t=t_for_model,
            alpha_mask=alpha_mask
        ).logits
        
        # normal CE loss calculation
        B, T, V = logits.shape

        safe_labels = labels.clone()
        safe_labels[labels == -100] = 0

        # Calculate accuracy
        preds = torch.argmax(logits, dim=-1) # (B, T)
        correct_mask = (preds == safe_labels) & p_mask_lm
        accuracy = correct_mask.sum().float() / p_mask_lm.sum().clamp(min=1).float()

        raw_loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100, reduction='none')
        loss_lm = (raw_loss * p_mask_lm).sum(dim=1)

        mask_num = p_mask_lm.sum(dim=1).clamp(min=1)
        loss_lm = loss_lm / mask_num  # shape: (B,)

        safe_t = torch.clamp(t.squeeze(-1), min=1e-5, max=1.0 - 1e-5)
        weight_shape = [-1] + [1] * (loss_lm.dim() - 1)
        if weighting_strategy == "constant": # no weighting, treat all noise levels equally
            loss_lm = (loss_lm).sum() / B          
        elif weighting_strategy == "symmetric": # peak at middle noise level and downweight both very low and very high noise levels
            weight = 4.0 * safe_t * (1.0 - safe_t)
            weight = weight.view(*weight_shape)
            loss_lm = (loss_lm * weight).sum() / B       
        elif weighting_strategy == "noise_focused": # upweight higher noise levels and downweight lower noise levels
            weight = safe_t
            weight = weight.view(*weight_shape)
            loss_lm = (loss_lm * weight).sum() / B
        elif weighting_strategy == "clean_focused": # upweight lower noise levels and downweight higher noise levels
            weight = 1 / safe_t
            weight = weight.view(*weight_shape)
            loss_lm = (loss_lm * weight).sum() / B
        else:
            raise ValueError(f"Unknown weighting_strategy: {weighting_strategy}")
        
        if accelerator.is_local_main_process:
            logger.info(f"flowmatch loss: {loss_lm.item()}")
            logger.info(f"flowmatch accuracy: {accuracy.item()}")
            
        return loss_lm, accuracy, logits


    # * ---- flow-match sampling logic ----
    def flowmatch_sampling(flow_start_probs, t, target_probs):
        """
        Calculates the intermediate state x_t between the prior and a continuous target distribution.
        """
        # 0. Convert to high precision for stability
        flow_start_probs = flow_start_probs.to(torch.float32)
        target_probs = target_probs.to(torch.float32)
        t = t.to(torch.float32)
        
        # 1. Map probability simplex to hypersphere
        x0 = torch.sqrt(flow_start_probs + 1e-10)
        x1 = torch.sqrt(target_probs + 1e-10)
        
        # 2. Optimized dot product for continuous hypersphere
        dot = (x0 * x1).sum(dim=-1, keepdim=True)
        dot = torch.clamp(dot, -1.0, 1.0 - 1e-6) 
        theta = torch.acos(dot)

        # 3. Calculate interpolation coefficients
        t_scaled = t.view(-1, 1, 1)
        sin_theta = torch.sin(theta)
        denom = sin_theta + 1e-10
        
        coeff_x1_slerp = torch.sin((1.0 - t_scaled) * theta) / denom
        coeff_x0_slerp = torch.sin(t_scaled * theta) / denom
        
        # Handle small angles (Lerp fallback) to prevent NaN
        small_angle_mask = (theta < 1e-4)
        coeff_x1 = torch.where(small_angle_mask, 1.0 - t_scaled, coeff_x1_slerp)
        coeff_x0 = torch.where(small_angle_mask, t_scaled, coeff_x0_slerp)
        
        # 4. Calculate xt using full continuous interpolation
        xt = x0 * coeff_x0 + x1 * coeff_x1
        
        # 5. Map back to probability space
        flow_t_probs = torch.pow(xt, 2)
        flow_t_probs = flow_t_probs / (flow_t_probs.sum(dim=-1, keepdim=True) + 1e-10)

        # Return to model dtype (bfloat16)
        return flow_t_probs.to(dtype=model.dtype, device=flow_start_probs.device)


    # * ---- training loop preparation ----
    # Counter for actual optimizer updates (Global Steps) is initialized before checkpoint loading
    if global_update_step is None:
        global_update_step = 0
    
    # get accurate vocab size
    unwrapped_model = accelerator.unwrap_model(model)
    if hasattr(unwrapped_model, "get_input_embeddings"):
        vocab_size = unwrapped_model.get_input_embeddings().weight.shape[0]
    elif hasattr(unwrapped_model, "transformer") and hasattr(unwrapped_model.transformer, "wte"):
        vocab_size = unwrapped_model.transformer.wte.weight.shape[0]
    else:
        # Fallback for dynamic fetching if specific attributes are hidden
        vocab_size = next(iter(unwrapped_model.parameters())).shape[0]
    
    # schedule temperature for flow matching teacher forcing
    curriculum_steps = max(1, max_train_steps * 0.5)
    temp_correct_start = config.training.temp_correct_start
    guidance_incorrect_start = config.training.guidance_incorrect_start


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
        
        flowmatch_loss_meter = AverageMeter()   
        flowmatch_acc_meter = AverageMeter()
        stage_loss_meters = [AverageMeter() for _ in range(config.training.num_sim_stages)]
        stage_acc_meters = [AverageMeter() for _ in range(config.training.num_sim_stages)]

        for step, batch in enumerate(progress_bar, start=1):
            with accelerator.accumulate(model):
                
                import contextlib

                # Prepare batch, move to device
                input_ids = batch["input_ids"].to(accelerator.device)
                labels    = batch["labels"].to(accelerator.device)
                p_mask_lm = batch["p_mask_lm"].to(accelerator.device)

                # Configure num_sim_stages and sample t for flow matching in this batch
                num_sim_stages = config.training.num_sim_stages
                sim_t_sampling = config.training.sim_t_sampling
                buckets = None
                sim_t_list = []
                if sim_t_sampling == "uniform":
                    buckets = torch.linspace(0.0, 1.0, num_sim_stages + 1, device=accelerator.device)
                elif sim_t_sampling == "clean_noise_focused":
                    # Use cosine spacing to densely sample near 0 and 1, while keeping it strictly bounded in [0, 1]
                    uniform_buckets = torch.linspace(0.0, 1.0, num_sim_stages + 1, device=accelerator.device)
                    buckets = (1.0 - torch.cos(uniform_buckets * torch.pi)) / 2.0
                else:
                    raise ValueError(f"Unknown sim_t_sampling: {sim_t_sampling}")
                for i in range(num_sim_stages):
                    # Sample uniformly within the current bucket for each sequence in the batch
                    # Shape of t_i: (B, 1)
                    t_i = torch.rand(input_ids.size(0), 1, device=input_ids.device) * (buckets[i+1] - buckets[i]) + buckets[i]
                    sim_t_list.append(t_i)
                t_batch = torch.stack(sim_t_list, dim=0)                 
                # Reverse the order so the simulation strictly follows the denoising trajectory: Noise (t~1) -> Clean (t~0)
                t_batch = torch.flip(t_batch, dims=[0])
                
                # Arrange the start probs of FM simulation and labels
                flow_start_probs = F.one_hot(input_ids, num_classes=vocab_size).to(dtype=model.dtype)
                curr_start_probs = flow_start_probs.clone()

                # The Simulation Loop
                batch_flowmatch_loss = 0.0
                batch_flowmatch_acc = 0.0
                batch_stage_losses = []
                batch_stage_accs = []
                for stage_idx in range(num_sim_stages):

                    # Set up time for current stage and not syncing gradients until the last stage
                    t_current = t_batch[stage_idx]
                    is_last_stage = (stage_idx == num_sim_stages - 1)
                    sync_context = contextlib.nullcontext() if is_last_stage else accelerator.no_sync(model)

                    with sync_context:
                        # Forward pass
                        stage_loss, stage_acc, stage_logits = flowmatch_forward_process(
                            input_ids=input_ids,
                            probs=curr_start_probs,
                            t=t_current,
                            labels=labels,
                            weighting_strategy=config.training.flowmatch_loss_weighting_strategy
                        )

                        # Scale the loss to prevent magnitude explosion across multiple steps
                        scaled_stage_loss = stage_loss / num_sim_stages

                        # Immediate backward to free the computation graph for this specific step
                        accelerator.backward(scaled_stage_loss)

                    # Compute probabilities for the next stage's sampling without tracking gradients
                    detached_logits = stage_logits.detach()

                    # Schedule temperature for teacher forcing based on global update step and curriculum steps
                    progress = min(global_update_step / curriculum_steps, 1.0)
                    Temp_correct = temp_correct_start + (1.0 - temp_correct_start) * progress # For sharpening the distribution of correct predictions
                    guidance_rate = guidance_incorrect_start + (0.0 - guidance_incorrect_start) * progress # For guiding the wrongly predicted positions

                    # Get predictions and correct mask
                    preds = torch.argmax(detached_logits, dim=-1) # (B, T)
                    correct_mask = (preds == labels) & p_mask_lm
                    incorrect_mask = (preds != labels) & p_mask_lm

                    # Apply specific temperatures to the correct positions
                    temp_scaler = torch.ones_like(detached_logits, dtype=torch.float32)
                    temp_scaler = torch.where(correct_mask.unsqueeze(-1), Temp_correct, temp_scaler)
                    base_probs = F.softmax(detached_logits.float() / temp_scaler, dim=-1).to(model.dtype) # prob after correct token sharpening

                    safe_labels = torch.where(labels != -100, labels, 0)
                    gt_one_hot = F.one_hot(safe_labels, num_classes=vocab_size).to(model.dtype)
                    guided_probs = (1.0 - guidance_rate) * base_probs + guidance_rate * gt_one_hot

                    curr_end_probs = torch.where(
                        incorrect_mask.unsqueeze(-1),
                        guided_probs,
                        base_probs
                    )

                    batch_flowmatch_loss += stage_loss.detach() / num_sim_stages
                    batch_flowmatch_acc += (stage_acc.detach() if isinstance(stage_acc, torch.Tensor) else stage_acc) / num_sim_stages
                    batch_stage_losses.append(stage_loss.detach())
                    batch_stage_accs.append(stage_acc.detach() if isinstance(stage_acc, torch.Tensor) else stage_acc)

                    if stage_idx < num_sim_stages - 1:
                        t_next = t_batch[stage_idx + 1]

                        with torch.no_grad():
                            curr_start_probs = flowmatch_sampling(
                                flow_start_probs=flow_start_probs,
                                t=t_next,
                                target_probs=curr_end_probs
                            )

                batch_stage_losses = torch.stack(batch_stage_losses).unsqueeze(0) # shape (1, num_sim_stages)
                batch_stage_accs = torch.stack(batch_stage_accs).unsqueeze(0) # shape (1, num_sim_stages)

                flowmatch_loss_gathered = accelerator.gather(batch_flowmatch_loss.unsqueeze(0)).mean()
                flowmatch_acc_gathered = accelerator.gather(batch_flowmatch_acc.unsqueeze(0)).mean()
                stage_loss_gathered = accelerator.gather(batch_stage_losses).mean(dim=0)
                stage_acc_gathered = accelerator.gather(batch_stage_accs).mean(dim=0)

                flowmatch_loss_meter.update(flowmatch_loss_gathered.item(), n=input_ids.size(0))
                flowmatch_acc_meter.update(flowmatch_acc_gathered.item(), n=input_ids.size(0))
                for stage_idx in range(num_sim_stages):
                    stage_loss_meters[stage_idx].update(stage_loss_gathered[stage_idx].item(), n=input_ids.size(0))
                    stage_acc_meters[stage_idx].update(stage_acc_gathered[stage_idx].item(), n=input_ids.size(0))

                # Gradient clipping and optimizer step only on the last stage after all gradients have been accumulated across stages
                if accelerator.sync_gradients:
                    if config.training.max_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                # Perform optimizer step
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    # Increment global update step and lr scheduler
                    lr_scheduler.step()
                    global_update_step += 1

                    if accelerator.is_local_main_process:
                        print(f"Global Step {global_update_step} | Flowmatch Loss: {flowmatch_loss_meter.avg:.4f} | Flowmatch Acc: {flowmatch_acc_meter.avg:.4f}")

                    log_dict = {
                        "loss": flowmatch_loss_meter.avg,
                        "flowmatch_loss": flowmatch_loss_meter.avg,
                        "flowmatch_accuracy": flowmatch_acc_meter.avg,
                        "lr": lr_scheduler.get_last_lr()[0]
                    }                    
                    for i in range(num_sim_stages):
                        log_dict[f"Stage_Loss/stage_{i}"] = stage_loss_meters[i].avg
                        log_dict[f"Stage_Accuracy/stage_{i}"] = stage_acc_meters[i].avg

                    accelerator.log(log_dict, step=global_update_step)
                    
                    flowmatch_loss_meter.reset()
                    flowmatch_acc_meter.reset()
                    for i in range(num_sim_stages):
                        stage_loss_meters[i].reset()
                        stage_acc_meters[i].reset()

            del input_ids, labels, p_mask_lm, t_batch, flow_start_probs, curr_start_probs, \
                batch_flowmatch_loss, batch_flowmatch_acc, flowmatch_loss_gathered, flowmatch_acc_gathered, \
                stage_logits, curr_end_probs, stage_loss, scaled_stage_loss, \
                batch_stage_losses, batch_stage_accs, \
                stage_loss_gathered, stage_acc_gathered
            
        # Save checkpoint at the end of each epoch
        output_dir = Path(config.experiment.output_dir) / f"checkpoint-epoch-{epoch+1}"
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

        accelerator.wait_for_everyone() # Ensure all processes have finished saving before starting next epoch or ending

    accelerator.end_training()


if __name__ == "__main__":
    main()
