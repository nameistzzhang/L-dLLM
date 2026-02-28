from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="INFO")


import torch
class UniversalPrompting():
    def __init__(self, text_tokenizer,
                 max_prompt_len=8000, max_gen_length=377, ignore_id=-100, post_num=32):
        """
        :param text_tokenizer: original text tokenizer
        """
        self.text_tokenizer = text_tokenizer
        self.max_gen_length = max_gen_length
        self.max_prompt_len = max_prompt_len
        self.ignore_id = ignore_id
        self.post_num = post_num

    # language modeling
    def lm_prompt(self, text_ids_pairs):
        prompts_tensor, responses_tensor = text_ids_pairs
        pad_id = self.text_tokenizer.pad_token_id

        input_ids = torch.cat([prompts_tensor, responses_tensor], dim=1)
        prompt_len = prompts_tensor.shape[1]

        max_seq_len = prompt_len + self.max_gen_length
        if input_ids.shape[1] > max_seq_len:
            input_ids = input_ids[:, :max_seq_len]

        labels = input_ids.clone()
        labels[:, :prompt_len] = self.ignore_id

        resp_labels = labels[:, prompt_len:]
        pad_mask = (resp_labels == pad_id)

        if self.post_num > 0:
            keep_pad_mask = pad_mask & (torch.cumsum(pad_mask.int(), dim=1) <= self.post_num)
            drop_pad_mask = pad_mask & ~keep_pad_mask
        else:
            drop_pad_mask = pad_mask
    
        resp_labels[drop_pad_mask] = self.ignore_id

        return input_ids, labels, prompt_len

    def mask_prompt(self):
        pass

    def __call__(self, input):
        prompts, responses = input

        enc = self.text_tokenizer(
            prompts,
            padding=False,
            truncation=False,
            return_length=True
        )
        lengths = enc["length"]
        # 2) 过滤出长度 <= max_len 的 indices
        keep_indices = [i for i, L in enumerate(lengths) if L <= self.max_prompt_len]
        drop_num = len(prompts) - len(keep_indices)
        
        prompts  = [prompts[i]  for i in keep_indices]
        responses = [responses[i] for i in keep_indices]

        # 使用 tokenizer 将 raw text 转为 token ids
        prompt_ids = self.text_tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            padding_side = "left"
        )['input_ids']
        response_ids = self.text_tokenizer(
            responses,
            padding=True,
            return_tensors="pt",
            padding_side = "right"
        )['input_ids']
        input_ids_lm, labels_lm, start_pos = self.lm_prompt((prompt_ids, response_ids))
        return input_ids_lm, labels_lm, start_pos, drop_num


if __name__ == '__main__':
    pass