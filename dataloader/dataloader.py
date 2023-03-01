import torch
import numpy as np
import string
from transformers import AutoTokenizer, AutoModel
import nltk
import string
from torch.utils import data
import nltk.stem
from nltk.corpus import stopwords
import nltk
chachedWords = stopwords.words('english')
nltk.download('punkt')
nltk.download("stopwords")
chachedWords = stopwords.words('english')
stop_words = set(stopwords.words("english"))

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)

class ReconScratchDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_file,
            dataset_mode,
            bow_vocab_size,
            vocab_list,
            ): 

        self.data_file = data_file # train dataset and test dataset
        self.dataset_mode = dataset_mode
        self.twitter_hashtag = []
        
        self.vocab_list = vocab_list
        self.vocab_count_dict = {}
        self.vocab_size = bow_vocab_size

        self.process_data_file()
    
    def __getitem__(self, idx): 
        twitter, hashtag = self.twitter_hashtag[idx]
        twitter_tokens = self.get_twitter_tokens(twitter)  # a token list with length of n
        hashtag_label = self.get_hashtag_label(hashtag)  # an int element converted to tensor
        bow_feature = self.get_bow_features(twitter)

        return twitter_tokens, torch.tensor(hashtag_label), bow_feature

    
    def get_bow_features(self, twitter):
        text = twitter.lower()
        remove = str.maketrans('', '',string.punctuation)
        without_punctuation = text.translate(remove)
        tokens = nltk.word_tokenize(without_punctuation)
        without_stopwords = [w for w in tokens if not w in stop_words]
        s = nltk.stem.SnowballStemmer('english')
        cleaned_text = [s.stem(ws) for ws in without_stopwords]
        self.vocab_list = self.vocab_list[0:self.vocab_size]
        bow_feature = [0.]*self.vocab_size

        for token in cleaned_text:
            try:
                oneIdx = self.vocab_list.index(token)
                bow_feature[oneIdx] = 1
            except:
                bow_feature[1] = 1  # UNK for index 1

        return bow_feature


    def get_twitter_tokens(self, twitter):
        # get word index vector from bert vocabulary
        idx_list = []
        lower = twitter.lower()
        remove = str.maketrans('', '', string.punctuation)
        without_punctuation = lower.translate(remove)
        tokens = nltk.word_tokenize(without_punctuation)
        
        return tokens

    def get_hashtag_label(self, hashtag):
        return int(hashtag) 

    def process_data_file(self):
        # process bert vocabulary and twitter dataset
        f = open(self.data_file)
        for line in f:
            l = line.strip('\n').split('\t')
            twitter, hashtag = str(l[2]).lower(), str(l[1]).lower()
            if len(twitter) > 0 and len(hashtag) > 0:
                self.twitter_hashtag.append((twitter, hashtag))

        # formulate vocab_dict
        if self.dataset_mode==True:
            for twitter, _ in self.twitter_hashtag:
                text = twitter.lower()
                remove = str.maketrans('', '',string.punctuation)
                without_punctuation = text.translate(remove)
                tokens = nltk.word_tokenize(without_punctuation)
                without_stopwords = [w for w in tokens if not w in stop_words]
                s = nltk.stem.SnowballStemmer('english')
                cleaned_text = [s.stem(ws) for ws in without_stopwords]
                
                for token in cleaned_text:
                    if len(token) > 1 and not(token.isdigit()):
                        try:
                            self.vocab_count_dict[token] += 1
                        except:
                            self.vocab_count_dict[token] = 1
                    
                #print(self.vocab_count_dict)
            list1= sorted(self.vocab_count_dict.items(),key=lambda x:x[1], reverse=True)
            test1 = list(dict(list1).keys())
            # test2 = list(dict(list1).items())
            print(test1[0:50])
            # print(test2)
            for token in test1:
                self.vocab_list.append(token)
    def get_vocab_list(self):
        return self.vocab_list

    # def get_vocab_size(self):
    #     self.vocab_size = len(self.vocab_list)
    #     return self.vocab_size
            
    def __len__(self):
        return len(self.twitter_hashtag)

def recon_my_collate(batch):
 
    batch_sentences = []
    batch_labels = []
    batch_bows = []
    for twitter_tokens, hashtag_label, bow_feature in batch:
        batch_sentences.append(twitter_tokens)
        batch_labels.append(hashtag_label)
        batch_bows.append(bow_feature)

    try:
        batch_dict = tokenizer(batch_sentences, is_split_into_words=True, truncation=True, padding=True, return_tensors='pt')

        input_ids, token_type_ids, attention_mask = batch_dict['input_ids'], batch_dict['token_type_ids'], batch_dict['attention_mask']
        
        return input_ids, token_type_ids, attention_mask, torch.tensor(batch_labels), torch.tensor(batch_bows)
    except:
        return 'Flag', len(batch_labels), 0, 0

class TimeScratchDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_file,
            dataset_mode,
            bow_vocab_size,
            vocab_list,
            ): 

        self.data_file = data_file # train dataset and test dataset
        self.dataset_mode = dataset_mode
        self.twitter_hashtag = []
        
        self.vocab_list = vocab_list
        self.vocab_count_dict = {}
        self.vocab_size = bow_vocab_size

        self.process_data_file()
    
    def __getitem__(self, idx): 
        twitter, hashtag = self.twitter_hashtag[idx]
        twitter_tokens = self.get_twitter_tokens(twitter)  # a token list with length of n
        hashtag_label = self.get_hashtag_label(hashtag)  # an int element converted to tensor
        bow_feature = self.get_bow_features(twitter)

        return twitter_tokens, torch.tensor(hashtag_label), bow_feature

    
    def get_bow_features(self, twitter):
        text = twitter.lower()
        remove = str.maketrans('', '',string.punctuation)
        without_punctuation = text.translate(remove)
        tokens = nltk.word_tokenize(without_punctuation)
        without_stopwords = [w for w in tokens if not w in stop_words]
        s = nltk.stem.SnowballStemmer('english')
        cleaned_text = [s.stem(ws) for ws in without_stopwords]
        self.vocab_list = self.vocab_list[0:self.vocab_size]
        bow_feature = [0.]*self.vocab_size

        for token in cleaned_text:
            try:
                oneIdx = self.vocab_list.index(token)
                bow_feature[oneIdx] = 1
            except:
                bow_feature[1] = 1  # UNK for index 1

        return bow_feature


    def get_twitter_tokens(self, twitter):
        # get word index vector from bert vocabulary
        idx_list = []
        lower = twitter.lower()
        remove = str.maketrans('', '', string.punctuation)
        without_punctuation = lower.translate(remove)
        tokens = nltk.word_tokenize(without_punctuation)
        
        return tokens

    def get_hashtag_label(self, hashtag):
        return int(hashtag) 

    def process_data_file(self):
        # process bert vocabulary and twitter dataset
        f = open(self.data_file)
        for line in f:
            l = line.strip('\n').split('\t')
            twitter, hashtag = str(l[2]).lower(), str(l[1]).lower()
            if len(twitter) > 0 and len(hashtag) > 0:
                self.twitter_hashtag.append((twitter, hashtag))

        # formulate vocab_dict
        if self.dataset_mode==True:
            for twitter, _ in self.twitter_hashtag:
                text = twitter.lower()
                remove = str.maketrans('', '',string.punctuation)
                without_punctuation = text.translate(remove)
                tokens = nltk.word_tokenize(without_punctuation)
                without_stopwords = [w for w in tokens if not w in stop_words]
                s = nltk.stem.SnowballStemmer('english')
                cleaned_text = [s.stem(ws) for ws in without_stopwords]
                
                for token in cleaned_text:
                    if len(token) > 1 and not(token.isdigit()):
                        try:
                            self.vocab_count_dict[token] += 1
                        except:
                            self.vocab_count_dict[token] = 1
                    
                #print(self.vocab_count_dict)
            list1= sorted(self.vocab_count_dict.items(),key=lambda x:x[1], reverse=True)
            test1 = list(dict(list1).keys())
            # test2 = list(dict(list1).items())
            print(test1[0:50])
            # print(test2)
            for token in test1:
                self.vocab_list.append(token)
    def get_vocab_list(self):
        return self.vocab_list

    # def get_vocab_size(self):
    #     self.vocab_size = len(self.vocab_list)
    #     return self.vocab_size
            
    def __len__(self):
        return len(self.twitter_hashtag)


def time_my_collate(batch):
 
    batch_sentences = []
    batch_labels = []
    batch_bows = []
    for twitter_tokens, hashtag_label, bow_feature in batch:
        batch_sentences.append(twitter_tokens)
        batch_labels.append(hashtag_label)
        batch_bows.append(bow_feature)

    try:
        batch_dict = tokenizer(batch_sentences, is_split_into_words=True, truncation=True, padding=True, return_tensors='pt')

        input_ids, token_type_ids, attention_mask = batch_dict['input_ids'], batch_dict['token_type_ids'], batch_dict['attention_mask']
        
        return input_ids, token_type_ids, attention_mask, torch.tensor(batch_labels), torch.tensor(batch_bows)
    except:
        return 'Flag', len(batch_labels), 0, 0



def load_recon_dataloader(dataset, path_to_data, train, batch_size, bow_vocab_size, bow_vocab_list):
    dataset = ReconScratchDataset(data_file=path_to_data, dataset_mode=train, vocab_size=bow_vocab_size, vocab_list=bow_vocab_list)
    data_length=dataset.__len__()

    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=recon_my_collate, num_workers=8)
    return dataloader, data_length, dataset.get_bow_vocab_list()

def load_time_dataloader(dataset, path_to_data, train, batch_size, bow_vocab_size, bow_vocab_list):
    dataset = TimeScratchDataset(data_file=path_to_data, dataset_mode=train, vocab_size=bow_vocab_size, vocab_list=bow_vocab_list)
    data_length=dataset.__len__()

    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=time_my_collate, num_workers=8)
    return dataloader, data_length, dataset.get_bow_vocab_list()
