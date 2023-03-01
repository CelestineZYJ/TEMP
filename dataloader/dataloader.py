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
        
        query_tokens=self.get_twitter_tokens(twitter[0])
        
        n_twitter_tokens=[]  # totally n=10 twitters in list for each twtter's token list
        for twi in twitter[1:]:
            twi_tokens=self.get_twitter_tokens(twi)  # a token list with length of m
            n_twitter_tokens.append(twi_tokens)
        
        hashtag_label = self.get_hashtag_label(hashtag)  # an int element converted to tensor
        
        query_bow_feature = self.get_query_bow_features(twitter[0]) # get the bow feature of the twitter[0] as past query
        n_1_bow_feature = self.get_n_1_bow_features(twitter[1:]) # get the n-1 twitters as future answers, integrate all in a single bow_feature

        return query_tokens, n_twitter_tokens, query_bow_feature, n_1_bow_feature, torch.tensor(hashtag_label)

    
    def get_n_1_bow_features(self, twitter):
        text=''
        for twi in twitter:  # integrate all the n-1 twitters' words in a row of bow feature
            text+=twi.lower()+' '
        
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
    
    def get_query_bow_features(self, twitter):
        text+=twitter.lower()
        
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


    def get_twitter_tokens(self, twi):
        # get word index vector from bert vocabulary
        idx_list = []
        lower = twi.lower()
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
            hashtag=int(l[0])
            twitter=[]
            for twi in l[1:]:
                if len(twi)==0:
                    twi='0'
                twitter.append(twi)

            self.twitter_hashtag.append((twitter, hashtag))

        # formulate vocab_dict
        if self.dataset_mode==True:
            for twitter, _ in self.twitter_hashtag:
                for twi in twitter:
                    text = twi.lower()
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
    batch_queries = []
    batch_n_1_sentences = []
    batch_query_bows = []
    batch_n_1_bows = []
    batch_labels = []
    for idx in range(0,9):
        batch_twi_sentences=[]
        for _, n_twitter_tokens, _, _, _ in batch:
            batch_twi_sentences.append(n_twitter_tokens[idx])  # totally number of batch_size(like 4, 128, ...) token_input_ids in batch_twi_sentences
        batch_n_1_sentences.append(batch_twi_sentences)   # totally n=10 batch_twi_senences in batch_n_sentences
            
    for query_tokens, _, query_bow_feature, n_1_bow_feature, hashtag_label in batch:
        batch_queries.append(query_tokens)
        batch_labels.append(hashtag_label)
        batch_query_bows.append(query_bow_feature)
        batch_n_1_bows.append(n_1_bow_feature)
    
    
    # get the tensor feature of each batch item
    batch_n_1_bert_items=[]
    for batch_twi_sentences in batch_n_1_sentences: # totally 10 twis
        twi_batch_dict=tokenizer(batch_twi_sentences, is_split_into_words=True, truncation=True, padding=True, return_tensors='pt')
        twi_input_ids, twi_token_type_ids, twi_attention_mask = twi_batch_dict['input_ids'], twi_batch_dict['token_type_ids'], twi_batch_dict['attention_mask']
        batch_n_1_bert_items.append([twi_input_ids, twi_token_type_ids, twi_attention_mask])
        
    batch_query_dict = tokenizer(batch_queries, is_split_into_words=True, truncation=True, padding=True, return_tensors='pt')
    input_ids, token_type_ids, attention_mask = batch_query_dict['input_ids'], batch_query_dict['token_type_ids'], batch_query_dict['attention_mask']
    batch_query_bert_items=[input_ids, token_type_ids, attention_mask ]
    return batch_query_bert_items, batch_n_1_bert_items, torch.tensor(batch_query_bows), torch.tensor(batch_n_1_bows), torch.tensor(batch_labels)


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
        query_tokens = self.get_twitter_tokens(twitter[0])  # a token list with length of n
        future_tokens = self.get_twitter_tokens(twitter[1])
        hashtag_label = self.get_hashtag_label(hashtag)  # an int element converted to tensor
        query_bow_feature  = self.get_bow_features(twitter[0])
        future_bow_feature =  self.get_bow_features(twitter[1])

        return query_tokens, future_tokens, query_bow_feature, future_bow_feature, torch.tensor(hashtag_label)
    
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
            query, future, hashtag = str(l[1]).lower(), str(l[2]).lower(), int(l[0])
            twitter = [query, future]
            if len(query) > 0 and len(future) > 0 and len(hashtag) > 0:
                self.twitter_hashtag.append((twitter, hashtag))

        # formulate vocab_dict
        if self.dataset_mode==True:
            for twitter, _ in self.twitter_hashtag:
                for twi in twitter:
                    text = twi.lower()
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
    batch_queries = []
    batch_futures = []
    batch_query_bows = []
    batch_future_bows = []
    batch_labels = []
    
    for query_tokens, future_tokens, query_bow_feature, future_bow_features, hashtag_label in batch:
        batch_queries.append(query_tokens)
        batch_futures.append(future_tokens)
        batch_query_bows.append(query_bow_feature)
        batch_future_bows.append(future_bow_features)
        batch_labels.append(hashtag_label)

    batch_query_dict = tokenizer(batch_queries, is_split_into_words=True, truncation=True, padding=True, return_tensors='pt')
    query_input_ids, query_token_type_ids, query_attention_mask = batch_query_dict['input_ids'], batch_query_dict['token_type_ids'], batch_query_dict['attention_mask']
    batch_query_bert_items=[query_input_ids, query_token_type_ids, query_attention_mask]
    
    batch_future_dict = tokenizer(batch_futures, is_split_into_words=True, truncation=True, padding=True, return_tensors='pt')
    future_input_ids, future_token_type_ids, future_attention_mask = batch_future_dict['input_ids'], batch_future_dict['token_type_ids'], batch_future_dict['attention_mask']
    batch_future_bert_items=[future_input_ids, future_token_type_ids, future_attention_mask]
    
    return batch_query_bert_items, batch_future_bert_items, torch.tensor(batch_query_bows), torch.tensor(batch_future_bows), torch.tensor(batch_labels)



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
