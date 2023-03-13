import torch.nn as nn
import torch
import torch.nn.functional as F
import os
#device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")


class DVAE(nn.Module):
    def __init__(self, bow_vocab_size=20000, topic_num=128, hidden_dim=512, l1_strength=0.01):
        super(DVAE, self).__init__()
        # print(topic_num) # 50
        self.input_dim = bow_vocab_size
        self.topic_num = topic_num
        self.fc11 = nn.Linear(self.input_dim, hidden_dim)
        self.fc12 = nn.Linear(hidden_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, topic_num)
        self.fc22 = nn.Linear(hidden_dim, topic_num)
        self.fcs = nn.Linear(self.input_dim, hidden_dim, bias=False)
        self.fcg1 = nn.Linear(topic_num, topic_num)
        self.fcg2 = nn.Linear(topic_num, topic_num)
        self.fcg3 = nn.Linear(topic_num, topic_num)
        self.fcg4 = nn.Linear(topic_num, topic_num)
        self.fcd1 = nn.Linear(topic_num, self.input_dim)
        self.l1_strength = torch.FloatTensor([l1_strength])#.to(device)
        # self.dropout = torch.nn.Dropout(p=0.05)

    def encode(self, x):
        e1 = F.relu(self.fc11(x))
        e1 = F.relu(self.fc12(e1))
        e1 = e1.add(self.fcs(x))
        # dropout_e1=self.dropout(e1)
        # e1=dropout_e1
        return self.fc21(e1), self.fc22(e1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def generate(self, h):
        h=F.normalize(h)
        g1 = torch.relu(self.fcg1(h))
        g1 = torch.relu(self.fcg2(g1))
        g1 = torch.relu(self.fcg3(g1))
        g1 = torch.relu(self.fcg4(g1))
        g1 = g1.add(h)
        return g1

    def decode(self, z):
        d1 = F.softmax(self.fcd1(z), dim=1)
        # print(d1) # torch.Size([64, 10000])
        return d1

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        # print(z) # torch.Size([1, 50])
        g = self.generate(z)
        # print(self.decode(g).size()) # torch.Size([64, 10000])

        return z, g, self.decode(g), mu, logvar

    def print_topic_words(self, vocab_dic, fn, n_top_words=10):
        # print(self.fcd1.weight.data.size()) # torch.Size([64, 10000])
        beta_exp = self.fcd1.weight.data.cpu().numpy().T
        logging.info("Writing to %s" % fn)
        fw = open(fn, 'w')
        # print(beta_exp.shape) (100, 10000)
        for k, beta_k in enumerate(beta_exp):
            topic_words = [vocab_dic[w_id] for w_id in np.argsort(beta_k)[:-n_top_words - 1:-1]]
            print('Topic {}: {}'.format(k, ' '.join(topic_words)))
            fw.write('{}\n'.format(' '.join(topic_words)))
        fw.close()

    def print_case_topic(self, vocab_dict, z, n_top_words=10):
        g = self.generate(z)
        dg = self.decode(g)
        dg = dg.cpu().numpy() # numpy (1, 10000)
        for i_dg in dg:
            topic_words = [vocab_dict[w_id] for w_id in np.argsort(i_dg)[:-n_top_words - 1:-1]]
        return topic_words
    
    def print_word_weight(self, vocab_dict, z):
        g = self.generate(z)
        dg = self.decode(g)
        dg = dg.cpu().numpy() # numpy (1, 10000)
        for i_dg in dg:
            return i_dg

    def get_topic_num(self):
        return self.topic_num
