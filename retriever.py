from torch.utils.data.sampler import BatchSampler
from tqdm import tqdm
import torch.utils.data as data
import torch
import os
import sys
import json
import numpy as np
import random
from utils.loss import vae_loss_function
from utils.loss import l1_penalty
from models.dynamicVAE import DVAE
from models.retrieve_mlp import MLP
from dataloader.dataloader import load_recon_dataloader
from dataloader.dataloader import load_time_dataloader
import torch.nn.functional as F
from collections import OrderedDict



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Get config file from command line arguments
if len(sys.argv) != 2:
    raise(RuntimeError("Wrong arguments, use python main.py <config_path>"))
config_path = sys.argv[1]

# Open config file
with open(config_path) as f:
    config = json.load(f)

if config["path_to_data"] == "":
    raise(RuntimeError("Path to data not specified. Modify path_to_data attribute in config to point to data."))

adataset = config["dataset"]

def fix_model(model):
    for param in model.parameters():
        param.requires_grad = False


def unfix_model(model):
    for param in model.parameters():
        param.requires_grad = True



def cla_twis():
    
    train_data_loader, _, train_bow_vocab_list = load_recon_dataloader(dataset = config["dataset"], 
                                                    path_to_data = config["path_to_data"]['recon_train'],
                                                    train = True,
                                                    batch_size= config["training"]["batch_size"],
                                                    bow_vocab_size = config["bow_vocab_size"],
                                                    bow_vocab_list=[]
                                                    )
    
    valid_data_loader, _, _ = load_time_dataloader(dataset = config["dataset"], 
                                                    path_to_data = config["path_to_data"]['time_valid'],
                                                    train = False,
                                                    batch_size= config["training"]["batch_size"],
                                                    bow_vocab_size = config["bow_vocab_size"],
                                                    bow_vocab_list=train_bow_vocab_list
                                                    )
                                                
    test_data_loader, _, _ = load_time_dataloader(dataset = config["dataset"], 
                                                    path_to_data = config["path_to_data"]['time_test'],
                                                    train = False,
                                                    batch_size= config["training"]["batch_size"],
                                                    bow_vocab_size = config["bow_vocab_size"],
                                                    bow_vocab_list=train_bow_vocab_list
                                                    )
    pool_data_loader, _, _ = load_time_dataloader(dataset = config["dataset"], 
                                                    path_to_data = config["path_to_data"]['pool_test'],
                                                    train = False,
                                                    batch_size= config["training"]["batch_size"],
                                                    bow_vocab_size = config["bow_vocab_size"],
                                                    bow_vocab_list=train_bow_vocab_list
                                                    )
    query_data_loader, _, _ = load_time_dataloader(dataset = config["dataset"], 
                                                    path_to_data = config["path_to_data"]['query_test'],
                                                    train = False,
                                                    batch_size= config["training"]["batch_size"],
                                                    bow_vocab_size = config["bow_vocab_size"],
                                                    bow_vocab_list=train_bow_vocab_list
                                                    )
    
    
    #model, criterion, optimizer
    ntm_model = DVAE()
    model = MLP(ntm_model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.000005) 
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=50, threshold=1e-4, min_lr=1e-4)

   
    model = model.to(device)
    ntm_model = ntm_model.to(device)
    
    # 5, 40
    epochs = [5, 20]  # 100, 35 ok lr=0.0005, 60, 35, lr=0.001 ok
    epochs=[0,0]
    #train the model
    epoch = epochs[0]

    MODELID = 0

    best_valid_loss = 1e10
    fix_model(model)
    unfix_model(model.get_vae_model())
    for epoch in range(epoch):
        model.train()
        total_loss = 0.
        for batch_query_bert_items, batch_future_bert_items, X_query_bows, Y_future_bows, cls_labels in tqdm(train_data_loader, desc='train for the '+str(epoch)+' epoch:'):
            X_query_bows=X_query_bows.detach().float().to(device)
            Y_future_bows=Y_future_bows.detach().float().to(device)
            
            cls_labels=cls_labels.to(device)
            
            query_batch_dict = {'input_ids': batch_query_bert_items[0].to(device), 'token_type_ids': batch_query_bert_items[1].to(device), 'attention_mask': batch_query_bert_items[2].to(device)}

            # train process-------------------------------------
            optimizer.zero_grad()

            # forward process-------------------------------------
            pred_labels, recon_batch, X_bow_feature, mu, logvar, _ = model(query_batch_dict, X_query_bows)


            # compute loss
            
            lossFunction = torch.nn.CrossEntropyLoss()
            
            mlp_loss = lossFunction(pred_labels, cls_labels)

            Y_bow_feature = F.normalize(Y_future_bows)
            vae_loss = vae_loss_function(recon_batch, Y_bow_feature, mu, logvar)
            vae_loss = vae_loss + model.get_vae_model().l1_strength.to(device) * l1_penalty(model.get_vae_model().fcd1.weight)

            # print('@'*100)
            # print(mlp_loss)
            # print(vae_loss)
            loss = mlp_loss + vae_loss/1000

            total_loss += (loss.item() * len(cls_labels))

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0, norm_type=2)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0, norm_type=2)
            optimizer.step()

        print('train loss: '+str(total_loss / len(train_data_loader)*10))

        #validation
        #########################################################################################################
        model.eval()
        total_loss = 0.
        for batch_query_bert_items, batch_future_bert_items, X_query_bows, Y_future_bows, cls_labels, _ in tqdm(valid_data_loader, desc='valid for the '+str(epoch)+' epoch:'):
            
            X_query_bows=X_query_bows.detach().float().to(device)
            Y_future_bows=Y_future_bows.detach().float().to(device)
            
            cls_labels=cls_labels.to(device)
            
            query_batch_dict = {'input_ids': batch_query_bert_items[0].to(device), 'token_type_ids': batch_query_bert_items[1].to(device), 'attention_mask': batch_query_bert_items[2].to(device)}


            # eval process-------------------------------------
            optimizer.zero_grad()

            # forward process-------------------------------------
            pred_labels, recon_batch, X_bow_feature, mu, logvar, _  = model(query_batch_dict, X_query_bows)

            # compute loss
           
            lossFunction = torch.nn.CrossEntropyLoss()
            
            mlp_loss = lossFunction(pred_labels, cls_labels)

            Y_bow_feature = F.normalize(Y_future_bows)
            vae_loss = vae_loss_function(recon_batch, Y_bow_feature, mu, logvar)
            vae_loss = vae_loss + model.get_vae_model().l1_strength.to(device) * l1_penalty(model.get_vae_model().fcd1.weight)

            # print('@'*100)
            # print(mlp_loss)
            # print(vae_loss)
            loss = mlp_loss + vae_loss/1000

            total_loss += (loss.item() * len(cls_labels))


        print('valid loss: '+str(total_loss / len(valid_data_loader)*10)+'\n')



    #####################################################################################
    # joint train vae and mlp
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000005) 
    epoch = epochs[1]
    #unfix_model(model.get_vae_model())
    unfix_model(model)
    # fix_model(model.get_vae_model())

    for epoch in range(epoch):
        model.train()
        total_loss = 0.
        for batch_query_bert_items, batch_future_bert_items, X_query_bows, Y_future_bows, cls_labels in tqdm(train_data_loader, desc='train for the '+str(epoch)+' epoch:'):
            X_query_bows=X_query_bows.detach().float().to(device)
            Y_future_bows=Y_future_bows.detach().float().to(device)
            
            cls_labels=cls_labels.to(device)
            
            query_batch_dict = {'input_ids': batch_query_bert_items[0].to(device), 'token_type_ids': batch_query_bert_items[1].to(device), 'attention_mask': batch_query_bert_items[2].to(device)}

            # train process-------------------------------------
            optimizer.zero_grad()

            # forward process-------------------------------------
            pred_labels, recon_batch, X_bow_feature, mu, logvar, _ = model(query_batch_dict, X_query_bows)


            # compute loss
            
            lossFunction = torch.nn.CrossEntropyLoss()
            
            mlp_loss = lossFunction(pred_labels, cls_labels)

            Y_bow_feature = F.normalize(Y_future_bows)
            vae_loss = vae_loss_function(recon_batch, Y_bow_feature, mu, logvar)
            vae_loss = vae_loss + model.get_vae_model().l1_strength.to(device) * l1_penalty(model.get_vae_model().fcd1.weight)

            # print('@'*100)
            # print(mlp_loss)
            # print(vae_loss)
            loss = mlp_loss + vae_loss/1000

            total_loss += (loss.item() * len(cls_labels))

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0, norm_type=2)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0, norm_type=2)
            optimizer.step()
        print('train loss: '+str(total_loss / len(train_data_loader)*10))
        
        #validation
        #########################################################################################################
        model.eval()
        total_loss = 0.
        for batch_query_bert_items, batch_future_bert_items, X_query_bows, Y_future_bows, cls_labels, _ in tqdm(valid_data_loader, desc='valid for the '+str(epoch)+' epoch:'):
            
            X_query_bows=X_query_bows.detach().float().to(device)
            Y_future_bows=Y_future_bows.detach().float().to(device)
            
            cls_labels=cls_labels.to(device)
            
            query_batch_dict = {'input_ids': batch_query_bert_items[0].to(device), 'token_type_ids': batch_query_bert_items[1].to(device), 'attention_mask': batch_query_bert_items[2].to(device)}


            # eval process-------------------------------------
            optimizer.zero_grad()

            # forward process-------------------------------------
            pred_labels, recon_batch, bow_feature, mu, logvar, _  = model(query_batch_dict, X_query_bows)

            # compute loss
           
            lossFunction = torch.nn.CrossEntropyLoss()
            
            mlp_loss = lossFunction(pred_labels, cls_labels)

            Y_bow_feature = F.normalize(Y_future_bows)
            vae_loss = vae_loss_function(recon_batch, Y_bow_feature, mu, logvar)
            vae_loss = vae_loss + model.get_vae_model().l1_strength.to(device) * l1_penalty(model.get_vae_model().fcd1.weight)

            # print('@'*100)
            # print(mlp_loss)
            # print(vae_loss)
            loss = mlp_loss + vae_loss/1000

            total_loss += (loss.item() * len(cls_labels))

        print('valid loss: '+str(total_loss / len(valid_data_loader)*10)+'\n')

        #torch.save(model.state_dict(), '/mnt/zyj/bertJSVStanceDetection/'+f'model_{epoch}.pt')
        
        torch.save(model.state_dict(), '/mnt/zyj/irTemp/retrieve/'+adataset+'/'+f'model_{epoch}.pt')
        if total_loss < best_valid_loss :
            best_valid_loss = total_loss
            best_epoch = epoch
            
            print('best_epoch!\n')
        ###################################################################################################
        
        with torch.no_grad():
            true_count = 0
            all_count = 0

            label0 = 0
            acclabel0 = 0
            label1 = 0
            acclabel1 = 0

            

            for batch_query_bert_items, batch_future_bert_items, X_query_bows, Y_future_bows, cls_labels, _ in tqdm(test_data_loader, desc='test for the model:'):

                X_query_bows=X_query_bows.detach().float().to(device)
                Y_future_bows=Y_future_bows.detach().float().to(device)
                
                cls_labels=cls_labels.to(device)
                
                query_batch_dict = {'input_ids': batch_query_bert_items[0].to(device), 'token_type_ids': batch_query_bert_items[1].to(device), 'attention_mask': batch_query_bert_items[2].to(device)}


                pred_labels, recon_batch, X_bow_feature, mu, logvar, _  = model(query_batch_dict, X_query_bows)
                
                
                # print('*'*100)
                pred_max_labels = pred_labels.argmax(dim=1)

                true = cls_labels.cpu().numpy()
                pred = pred_max_labels.cpu().numpy()

                true_list = true.tolist()
                pred_list = pred.tolist()

                acc = (true - pred).tolist()

                ###################################################################################################################
                true_list = true.tolist()
                pred_list = pred.tolist()


                for idx, label in enumerate(true_list):
                    if label == 0:
                        label0 += 1
                        if pred_list[idx] == 0:
                            acclabel0 += 1
                    elif label == 1:
                        label1 += 1
                        if pred_list[idx] == 1:
                            acclabel1 += 1


                ###################################################################################################################

                # print(acc)
                
                for i in acc:
                    all_count += 1
                    if i == 0:
                        true_count += 1
                # except:
                #     pass

            print('label 0: '+str(label0))
            print('acc label 0: '+str(float(acclabel0/label0)))

            print('label 1: '+str(label1))
            print('acc label 1: '+str(float(acclabel1/label1)))



            print('true count: '+str(true_count))
            print('all count: '+str(all_count))
            print(float(true_count/all_count))


    # test the model
    if epoch != 0:
        MODELID = best_epoch
    MODELID=1
    #########################################################################################################################################
    model.load_state_dict(torch.load('/mnt/zyj/irTemp/retrieve/'+adataset+'/'+f'model_{MODELID}.pt')) # 190!! 210!!450!!
    #########################################################################################################################################
    
    # re-ranking
    '''
    query_features = []
    pool_features = []
    with torch.no_grad():
        true_count = 0
        all_count = 0

        label0 = 0
        acclabel0 = 0
        label1 = 0
        acclabel1 = 0

        query_f = open(config["path_to_data"]['query_test'], 'r')
        query_lines = query_f.readlines()
        query_sen = OrderedDict()
        idx = 0 
        print(len(query_lines))
        print(len(query_data_loader))
        
        for batch_query_bert_items, batch_future_bert_items, X_query_bows, Y_future_bows, cls_labels, _ in tqdm(query_data_loader, desc='test for the model:'):

            X_query_bows=X_query_bows.detach().float().to(device)
            Y_future_bows=Y_future_bows.detach().float().to(device)
            
            cls_labels=cls_labels.to(device)
            
            query_batch_dict = {'input_ids': batch_query_bert_items[0].to(device), 'token_type_ids': batch_query_bert_items[1].to(device), 'attention_mask': batch_query_bert_items[2].to(device)}


            pred_labels, recon_batch, X_bow_feature, mu, logvar, dynamic_topic_dense_feature  = model(query_batch_dict, X_query_bows)
            
            dynamic_topic_dense_feature = dynamic_topic_dense_feature.cpu().numpy().tolist()
            
            for num in dynamic_topic_dense_feature:
                query_features.append(num)
                line = query_lines[idx].strip('\n').split('\t')
                label, sen = line[0], line[2]
                query_sen[label+'\t'+sen] = num
                idx += 1
                
 
        print(len(query_features))
        json_str = json.dumps(query_sen)
        with open(config["path_to_data"]["query_feature_dict"], 'w') as json_file:
            json_file.write(json_str)
    
            
    with torch.no_grad():
        true_count = 0
        all_count = 0

        label0 = 0
        acclabel0 = 0
        label1 = 0
        acclabel1 = 0

        pool_f = open(config["path_to_data"]['pool_test'], 'r')
        pool_lines = pool_f.readlines()

        pool_sen = OrderedDict()
        idx = 0 
        print(len(pool_lines))
        print(len(pool_data_loader))
        
        for batch_query_bert_items, batch_future_bert_items, X_query_bows, Y_future_bows, cls_labels, _ in tqdm(pool_data_loader, desc='test for the model:'):

            X_query_bows=X_query_bows.detach().float().to(device)
            Y_future_bows=Y_future_bows.detach().float().to(device)
            
            cls_labels=cls_labels.to(device)
            
            query_batch_dict = {'input_ids': batch_query_bert_items[0].to(device), 'token_type_ids': batch_query_bert_items[1].to(device), 'attention_mask': batch_query_bert_items[2].to(device)}


            pred_labels, recon_batch, X_bow_feature, mu, logvar, dynamic_topic_dense_feature  = model(query_batch_dict, X_query_bows)
            
            dynamic_topic_dense_feature = dynamic_topic_dense_feature.cpu().numpy().tolist()
            
            for num in dynamic_topic_dense_feature:
                pool_features.append(num)
                line = pool_lines[idx].strip('\n').split('\t')
                label, sen = line[0], line[2]
                pool_sen[sen] = num
                idx += 1
                
        
        print(len(pool_features))
        json_str = json.dumps(pool_sen)
        with open(config["path_to_data"]["pool_feature_dict"], 'w') as json_file:
            json_file.write(json_str)
    
    '''
    
    
    with open(config["path_to_data"]["query_feature_dict"], 'r', encoding='utf-8') as query_f:
        query_sen = json.load(query_f)
        
    with open(config["path_to_data"]["pool_feature_dict"], 'r') as pool_f:
        pool_sen = json.load(pool_f)
        
        
    query_f = open(config["path_to_data"]['query_test'], 'r')
    query_lines = query_f.readlines()
    
    pool_f = open(config["path_to_data"]['pool_test'], 'r')
    pool_lines = pool_f.readlines()
    
    # query_pool_rank = OrderedDict()
    # each_query_pool_rank = {}
        
    
    
    # for line in tqdm(query_lines):
    #     line = line.strip('\n').split('\t')
    #     label, sen = line[0], line[2]
    #     query_sentence = label+'\t'+sen
        
    #     for pool_line in (pool_lines):
    #         line = pool_line.strip('\n').split('\t')
    #         answer_sentence = line[2]
    #         each_query_dot = np.dot(np.array(query_sen[query_sentence]), np.array(pool_sen[answer_sentence]))
    #         each_query_pool_rank[answer_sentence] = each_query_dot
            
    #     # re-rank all the answers for each query by dot product value
    #     sort_each_query_pool_rank = sorted(each_query_pool_rank.items(), reverse=True, key=lambda item:item[1])
        
    #     query_pool_rank[query_sentence] = sort_each_query_pool_rank[:30]
    
    query_sentence_feature_matrix = np.zeros((len(query_lines),896))
    query_sentence_content_list = []
    answer_sentence_feature_matrix = np.zeros((len(pool_lines),896))
    answer_sentence_content_list = []
    for i in range(len(query_lines)):
        query_line = query_lines[i]
        line = query_line.strip('\n').split('\t')
        label, sen = line[0], line[2]
        query_sentence = label+'\t'+sen
        query_sentence_feature_matrix[i] = np.array(query_sen[query_sentence])
        query_sentence_content_list.append(query_sentence)

    for i in range(len(pool_lines)):
        pool_line = pool_lines[i]
        line = pool_line.strip('\n').split('\t')
        answer_sentence = line[2]
        answer_sentence_feature_matrix[i] = np.array(pool_sen[answer_sentence])
        answer_sentence_content_list.append(answer_sentence)

    query_dot_matrix = np.dot(query_sentence_feature_matrix,answer_sentence_feature_matrix.T)
    query_rank_matrix = np.argsort(-query_dot_matrix) # 由于np.argsort默认从小到大排序，因此query_dot_matrix取负号，就达到了从大到小排序的目的

    query_pool_rank = OrderedDict()
    for i in range(query_sentence_feature_matrix.shape[0]):
        result = []
        for j in range(30):
            result.append((answer_sentence_content_list[query_rank_matrix[i][j]],query_dot_matrix[i][query_rank_matrix[i][j]]))
        query_pool_rank[query_sentence_content_list[i]] = result
    json_str = json.dumps(query_pool_rank)
    with open(config["path_to_data"]["rerank_30"], 'w') as json_file:
        json_file.write(json_str)
    
    
    
        
    # with open(config["path_to_data"]["rerank_30"], 'r', encoding='utf-8') as json_file:
    #     query_pool_rank = json.load(json_file)
    
    
    
    # query_f = open(config["path_to_data"]['query_test'], 'r')
    # query_lines = query_f.readlines()[:2]
    
    # for line in tqdm(query_lines):
    #     line = line.strip('\n').split('\t')
    #     label, sen = line[0], line[2]
    #     query_sentence = label+'\t'+sen
    #     top10s = query_pool_rank[query_sentence]
    #     print(top10s[0])
            
            
cla_twis()
