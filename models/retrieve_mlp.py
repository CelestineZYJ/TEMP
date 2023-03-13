import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, NTM, input_size=768, hidden_size=2048):
        super(MLP, self).__init__()
        self.NTM = NTM
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.bertmodel = AutoModel.from_pretrained("bert-base-uncased")
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        #cself.bn1 = torch.nn.BatchNorm1d(num_features = self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size+self.NTM.get_topic_num(), 3)
        #self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, batch_dict, bow_feature):
        input_ids, token_type_ids, attention_mask = batch_dict["input_ids"], batch_dict["token_type_ids"], batch_dict["attention_mask"]
        x = self.twitter_embedding(input_ids, token_type_ids, attention_mask)
        x = x[1]  # tensor(batch_size, 768)
        x = self.relu((self.fc1(x)))
        #x = self.relu(self.bn1(self.fc1(x)))
        bow_norm = F.normalize(bow_feature) 
        _, z_feature, recon_batch, mu, logvar = self.NTM(bow_norm)
        
        # print(x)
        # print(z_feature)
        # print('\n\n')
        x = torch.cat((x, z_feature*0.1), dim=1)   


        output = self.fc2(x)  # output: torch.Size([128, 50])
        #output = self.softmax(output)  # output: torch.Size([128, 50])
        return output, recon_batch, bow_feature, mu, logvar


    def twitter_embedding(self, input_ids, token_type_ids, attention_mask):
        batch_dict = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
        twitter_embedding = self.bertmodel(**batch_dict)
        return twitter_embedding


    def get_vae_model(self):
        return self.NTM
    
    
