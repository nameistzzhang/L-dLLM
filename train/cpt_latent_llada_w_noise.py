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


class AccuracyBucketMeter:
    """Track accuracy distribution across t (noise level) buckets: [0.8-1.0], [0.6-0.8], [0.4-0.6], [0.2-0.4], [0.0-0.2]"""
    def __init__(self):
        self.buckets = {
            "t_[0.8-1.0]": 0.0,
            "t_[0.6-0.8]": 0.0,
            "t_[0.4-0.6]": 0.0,
            "t_[0.2-0.4]": 0.0,
            "t_[0.0-0.2]": 0.0,
        }
        self.counts = {k: 0 for k in self.buckets}
    
    def update(self, accuracy_batch, t_batch):
        """
        accuracy_batch: torch.Tensor of shape (B,) 
        t_batch: torch.Tensor of shape (B, 1) or (B,) with t values in [0, 1]
        """
        acc_values = accuracy_batch.flatten().cpu().numpy()
        t_values = t_batch.flatten().cpu().numpy()
        
        for acc, t in zip(acc_values, t_values):
            if t >= 0.8:
                self.buckets["t_[0.8-1.0]"] += acc
                self.counts["t_[0.8-1.0]"] += 1
            elif t >= 0.6:
                self.buckets["t_[0.6-0.8]"] += acc
                self.counts["t_[0.6-0.8]"] += 1
            elif t >= 0.4:
                self.buckets["t_[0.4-0.6]"] += acc
                self.counts["t_[0.4-0.6]"] += 1
            elif t >= 0.2:
                self.buckets["t_[0.2-0.4]"] += acc
                self.counts["t_[0.2-0.4]"] += 1
            else:
                self.buckets["t_[0.0-0.2]"] += acc
                self.counts["t_[0.0-0.2]"] += 1
    
    def get_avg_buckets(self):
        avg_buckets = {}
        for bucket_name in self.buckets:
            if self.counts[bucket_name] > 0:
                avg_buckets[bucket_name] = self.buckets[bucket_name] / self.counts[bucket_name]
            else:
                avg_buckets[bucket_name] = 0.0
        return avg_buckets
    
    def get_overall_avg(self):
        """Return the true weighted average across all buckets"""
        total_sum = sum(self.buckets.values())
        total_count = sum(self.counts.values())
        return total_sum / total_count if total_count > 0 else 0.0
    
    def reset(self):
        self.buckets = {k: 0.0 for k in self.buckets}
        self.counts = {k: 0 for k in self.counts}



class TrainDataset(Dataset):
    def __init__(self, inputs, labels, pmasks, config):
        self.inputs = inputs
        self.labels = labels
        self.pmasks = pmasks
        self.config = config

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # get random t for flow matching sampling
        lower = self.config.training.lower_p
        upper = self.config.training.upper_p
        t = torch.rand(1) * (upper - lower) + lower    
        return (
            self.inputs[idx],
            self.labels[idx],
            self.pmasks[idx],
            t,
        )



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
    

    # * ---- util function for collating batches ----
    def simple_collate(batch):
        inp, lbl, msk, t = zip(*batch)  
        return {
            "input_ids":  torch.stack(inp),
            "labels":     torch.stack(lbl),
            "p_mask_lm":  torch.stack(msk),
            "t":          torch.stack(t)
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
    input_ids, labels, p_mask_lm, start_pos, drop_num= prepare_all_noisy_batch(prompt_list, response_list, step_map_list)
    dataset_lm = TrainDataset(input_ids, labels, p_mask_lm, config) # create dataset for language modeling with noisy inputs and corresponding labels and masks

    train_dataloader_lm = DataLoader(
        dataset_lm,
        batch_size=config.training.batch_size_lm,
        shuffle=True,
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
    def unnmask_forward_process(input_ids, labels, p_mask_lm, top_k=256):
        
        logits = model(input_ids).logits
        B, T, V = logits.shape

        safe_labels = labels.clone()
        safe_labels[labels == -100] = 0

        # Calculate accuracy
        preds = torch.argmax(logits, dim=-1) # (B, T)
        correct_mask = (preds == safe_labels) & p_mask_lm
        accuracy = correct_mask.sum().float() / p_mask_lm.sum().clamp(min=1).float()

        # Calculate topk accuracy (whether GT is in top-k predictions)
        topk = top_k
        _, topk_preds = torch.topk(logits, k=min(topk, V), dim=-1)  # (B, T, topk)
        gt_in_topk = (topk_preds == safe_labels.unsqueeze(-1)).any(dim=-1)  # (B, T)
        topk_acc_mask = gt_in_topk & p_mask_lm
        topk_accuracy = topk_acc_mask.sum().float() / p_mask_lm.sum().clamp(min=1).float()

        raw_loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100, reduction='none')
        loss_lm = (raw_loss * p_mask_lm).sum(dim=1)

        mask_num = p_mask_lm.sum(dim=1).clamp(min=1)
        loss_lm = (loss_lm / mask_num).sum() / B
        
        if accelerator.is_local_main_process:
            logger.info(f"unnmask loss: {loss_lm.item()}")
            logger.info(f"unnmask accuracy: {accuracy.item()}")
            logger.info(f"unnmask topk@{topk} accuracy: {topk_accuracy.item()}")
            
        return loss_lm, accuracy, topk_accuracy, logits
    
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
            accuracy_per_seq (torch.Tensor): Per-sequence accuracy for bucketing.
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

        # Calculate accuracy per sequence for bucketing
        preds = torch.argmax(logits, dim=-1) # (B, T)
        correct_mask = (preds == safe_labels) & p_mask_lm
        # Sum correct predictions per sequence, then divide by number of valid positions per sequence
        correct_per_seq = correct_mask.sum(dim=1)  # (B,)
        valid_per_seq = p_mask_lm.sum(dim=1).clamp(min=1)  # (B,)
        accuracy_per_seq = correct_per_seq.float() / valid_per_seq.float()  # (B,)

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
            logger.info(f"flowmatch mean accuracy: {accuracy_per_seq.mean().item()}")
            
        return loss_lm, accuracy_per_seq


    # * ---- flow-match sampling logic ----
    def flowmatch_sampling(unmask_logits, t, labels, sigma_max=0.1, top_k=256):

        # 0. Preparation
        safe_labels = labels.clone()
        mask_ignore = (labels == -100)
        safe_labels[mask_ignore] = 0
        t = t.to(torch.float32)
        
        # 1. get top-k indices
        _, topk_indices = torch.topk(unmask_logits, k=top_k, dim=-1) 
        
        # 2. add gt index to the candidate set, ensuring GT is always included in the small subset of K+1 dimensions
        gt_indices = safe_labels.unsqueeze(-1)
        active_indices = torch.cat([topk_indices, gt_indices], dim=-1) 
        
        # 3. take only the k+1 logits for the active indices, resulting in a small tensor of shape [B, L, K+1]
        active_logits = unmask_logits.gather(dim=-1, index=active_indices).to(torch.float32)
        
        # 4. prevent duplicate indices (where GT is already in top-k) from being selected as noise by masking them with -inf.
        mask_duplicate = (topk_indices == gt_indices)
        mask_duplicate = torch.cat([mask_duplicate, torch.zeros_like(mask_duplicate[..., :1])], dim=-1)
        active_logits.masked_fill_(mask_duplicate, float('-inf'))
        
        # 5. flow matching with noise
        flow_start_probs = F.softmax(active_logits, dim=-1)
        
        # 5.0 Map to hypersphere
        x0 = torch.sqrt(flow_start_probs + 1e-10)
        x0.masked_fill_(mask_duplicate, 0.0)

        # 5.1 GT is the last dimension in this micro space, so the dot product with GT is just the last dimension of x0
        x0_target = x0[..., -1:]
        dot = torch.clamp(x0_target, -1.0, 1.0 - 1e-6) 
        theta = torch.acos(dot)

        # 5.2 Calculate interpolation coefficients
        t_scaled = t.view(-1, 1, 1)
        sin_theta = torch.sin(theta)
        denom = sin_theta + 1e-10
        
        coeff_x1_slerp = torch.sin((1.0 - t_scaled) * theta) / denom
        coeff_x0_slerp = torch.sin(t_scaled * theta) / denom
        
        small_angle_mask = (theta < 1e-4)
        coeff_x1 = torch.where(small_angle_mask, 1.0 - t_scaled, coeff_x1_slerp)
        coeff_x0 = torch.where(small_angle_mask, t_scaled, coeff_x0_slerp)
        
        # 5.3 Calculate xt
        xt = x0 * coeff_x0
        # x1 is [0, 0, ..., 1], so we can directly add coeff_x1 to the last dimension of xt
        xt[..., -1:] = xt[..., -1:] + coeff_x1
        
        # 5.4 Add noise on tangent space
        safe_t = torch.clamp(t.squeeze(-1), min=1e-5, max=1.0 - 1e-5)
        sigma = sigma_max * 4.0 * safe_t * (1.0 - safe_t) 
        
        noise = torch.randn_like(xt) # [B, L, K+1]
        noise.masked_fill_(mask_duplicate, 0.0)
        dot_product = (noise * xt).sum(dim=-1, keepdim=True)
        noise_tangent = noise - dot_product * xt
        
        sigma_expanded = sigma.view(-1, 1, 1)
        xt_noisy = xt + sigma_expanded * noise_tangent
        xt_noisy = xt_noisy / (torch.norm(xt_noisy, dim=-1, keepdim=True) + 1e-10)
        
        # Map back to probability space
        flow_t_probs_small = torch.pow(xt_noisy, 2)
        flow_t_probs_small.masked_fill_(mask_duplicate, 0.0)
        flow_t_probs_small = flow_t_probs_small / (flow_t_probs_small.sum(dim=-1, keepdim=True) + 1e-10)

        # 6. safely refill with probabilities
        flow_t_probs = torch.zeros_like(unmask_logits)
        flow_t_probs.scatter_add_(
            dim=-1, 
            index=active_indices, 
            src=flow_t_probs_small.to(dtype=unmask_logits.dtype)
        )

        return flow_t_probs
    

    # * ---- training loop ----
    # Counter for actual optimizer updates (Global Steps) is initialized before checkpoint loading
    if global_update_step is None:
        global_update_step = 0
    
    for epoch in range(first_epoch, num_train_epochs):
        
        model.train()
        
        progress_bar = tqdm(
            train_dataloader_lm,
            desc=f"Epoch {epoch+1}/{num_train_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,    
            leave=True          
        )
        
        unmask_loss_meter = AverageMeter()
        flowmatch_loss_meter = AverageMeter()   
        unmask_acc_meter = AverageMeter()
        unmask_topk_acc_meter = AverageMeter()
        flowmatch_acc_bucket_meter = AccuracyBucketMeter()

        for step, batch in enumerate(progress_bar, start=1):
            with accelerator.accumulate(model):
                # Count total micro-steps processed
                
                input_ids = batch["input_ids"].to(accelerator.device)
                labels    = batch["labels"].to(accelerator.device)
                p_mask_lm = batch["p_mask_lm"].to(accelerator.device)
                t_batch   = batch["t"].to(accelerator.device)

                # Accumulate gradients manually
                unmask_loss_lm, unmask_acc, unmask_topk_acc, unmask_logits = unnmask_forward_process(
                    input_ids=input_ids,
                    labels=labels,
                    p_mask_lm=p_mask_lm,
                    top_k=config.training.top_k
                )

                # Calculate the probability after unmasking forward as starting point for flow matching forward
                flow_t_probs = flowmatch_sampling(unmask_logits.detach(), t_batch, labels, sigma_max=config.training.sigma_max, top_k=config.training.top_k)
                
                flowmatch_loss_lm, flowmatch_acc_per_seq = flowmatch_forward_process(
                    input_ids=input_ids,
                    probs=flow_t_probs,
                    t=t_batch,
                    labels=labels,
                    weighting_strategy=config.training.flowmatch_loss_weighting_strategy
                )

                # Update meters
                unmask_loss_gathered = accelerator.gather_for_metrics(unmask_loss_lm).mean()
                flowmatch_loss_gathered = accelerator.gather_for_metrics(flowmatch_loss_lm).mean()
                unmask_acc_gathered = accelerator.gather_for_metrics(unmask_acc).mean()
                unmask_topk_acc_gathered = accelerator.gather_for_metrics(unmask_topk_acc).mean()
                flowmatch_acc_per_seq_gathered = accelerator.gather_for_metrics(flowmatch_acc_per_seq)
                t_batch_gathered = accelerator.gather_for_metrics(t_batch)
                
                unmask_loss_meter.update(unmask_loss_gathered.item())
                flowmatch_loss_meter.update(flowmatch_loss_gathered.item())
                unmask_acc_meter.update(unmask_acc_gathered.item())
                unmask_topk_acc_meter.update(unmask_topk_acc_gathered.item())
                flowmatch_acc_bucket_meter.update(flowmatch_acc_per_seq_gathered, t_batch_gathered)

                # combine losses
                loss_lm = unmask_loss_lm + flowmatch_loss_lm
                
                accelerator.backward(loss_lm)

                # Perform optimizer step and lr scheduler step [accelerator automatically handles gradient synchronization and accumulation based on the configuration]
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    # Increment global update step and lr scheduler
                    lr_scheduler.step()
                    global_update_step += 1

                    if config.training.max_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(),
                                                    config.training.max_grad_norm)

                    if accelerator.is_local_main_process:
                        print(f"Global Step {global_update_step} | Unmask Loss: {unmask_loss_meter.avg:.4f} | Flowmatch Loss: {flowmatch_loss_meter.avg:.4f} | Unmask Acc: {unmask_acc_meter.avg:.4f} | Unmask Top-k Acc: {unmask_topk_acc_meter.avg:.4f}")

                    flowmatch_bucket_avgs = flowmatch_acc_bucket_meter.get_avg_buckets()
                    
                    log_dict = {
                        "loss": unmask_loss_meter.avg + flowmatch_loss_meter.avg,
                        "unmask_loss": unmask_loss_meter.avg,
                        "flowmatch_loss": flowmatch_loss_meter.avg,
                        "unmask_accuracy": unmask_acc_meter.avg,
                        "unmask_topk_accuracy": unmask_topk_acc_meter.avg,
                        "flowmatch_mean_accuracy": flowmatch_acc_bucket_meter.get_overall_avg(),
                        "lr": lr_scheduler.get_last_lr()[0]
                    }
                    
                    # Add bucketed accuracy metrics
                    for bucket_name, avg_acc in flowmatch_bucket_avgs.items():
                        log_dict[f"flowmatch_acc_{bucket_name}"] = avg_acc
                    
                    accelerator.log(log_dict, step=global_update_step)
                    
                    unmask_loss_meter.reset()
                    flowmatch_loss_meter.reset()
                    unmask_acc_meter.reset()
                    unmask_topk_acc_meter.reset()
                    flowmatch_acc_bucket_meter.reset()

            del input_ids, labels, p_mask_lm, loss_lm # release memory

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
