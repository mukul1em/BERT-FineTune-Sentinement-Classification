from transformers import BertTokenizer

MAX_LENGTH = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 1
ACCUMULATION = 2
BERT_PATH = 'bert-base-uncased'
MODEL_PATH = './model.bin'
TRAINING_FILE = '../input/imdb.csv'
TOKENIZER = BertTokenizer.from_pretrained(
                                            BERT_PATH, 
                                            do_lower_case=True)


