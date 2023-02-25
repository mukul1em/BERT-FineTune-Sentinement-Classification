import config
import torch

class BERTDataset:
    def __init__(self, review, target):
        self.review = review
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_length = config.MAX_LENGTH

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review)
        review = " ".join(review.split())
        inputs = self.tokenizer.encode_plus(
                review,
                None, 
                add_special_tokens = True,
                max_length = self.max_length,
                pad_to_max_length=True
                )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {
            'input_ids' : torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.target[item], dtype=torch.float)
        }

        



