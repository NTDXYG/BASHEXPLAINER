import pickle
import numpy as np
import faiss
import torch
import textdistance

from transformers import RobertaTokenizer, RobertaModel

from IR.bert_whitening import sents_to_vecs, transform_and_normalize


tokenizer = RobertaTokenizer.from_pretrained("D:\\codebert-base")
model = RobertaModel.from_pretrained("D:\\codebert-base")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(DEVICE)


# Calculating lexical similarity
def sim_edit(s1,s2):
    score=textdistance.levenshtein.normalized_similarity(s1,s2)  # levenshtein distance
    sim=score
    return sim


class Retrieval(object):
    def __init__(self, dim, whitening_file, kernel_file, bias_file, train_code_list, train_nl_list, Is_train):
        f = open(whitening_file, 'rb')
        self.bert_vec = pickle.load(f)
        f.close()
        f = open(kernel_file, 'rb')
        self.kernel = pickle.load(f)
        f.close()
        f = open(bias_file, 'rb')
        self.bias = pickle.load(f)
        f.close()

        self.dim = dim
        self.train_code_list, self.train_nl_list = train_code_list, train_nl_list
        self.id2text = None
        self.vecs = None
        self.ids = None
        self.index = None
        self.Is_train=Is_train

    def encode_file(self):
        all_texts = []
        all_ids = []
        all_vecs = []
        for i in range(len(self.train_code_list)):
            all_texts.append(self.train_code_list[i])
            all_ids.append(i)
            all_vecs.append(self.bert_vec[i].reshape(1,-1))
        all_vecs = np.concatenate(all_vecs, 0)
        id2text = {idx: text for idx, text in zip(all_ids, all_texts)}
        self.id2text = id2text
        self.vecs = np.array(all_vecs, dtype="float32")
        self.ids = np.array(all_ids, dtype="int64")

    def build_index(self, n_list):
        quant = faiss.IndexFlatIP(self.dim)
        index = faiss.IndexIVFFlat(quant, self.dim, min(n_list, self.vecs.shape[0]))
        index.train(self.vecs)
        index.add_with_ids(self.vecs, self.ids)
        self.index = index

    def single_query(self, code, topK):
        body = sents_to_vecs([code], tokenizer, model, ir=True)
        body = transform_and_normalize(body, self.kernel, self.bias)
        vec = body[[0]].reshape(1, -1).astype('float32')

        _, sim_idx = self.index.search(vec, topK)
        sim_idx = sim_idx[0].tolist()
        max_score = 0
        max_idx = 0
        code_score_list = []
        for j in sim_idx:
            code_score=sim_edit(self.train_code_list[j].split(), code.split())
            code_score_list.append(code_score)
        if self.Is_train==False:     #Test phase: retrieve the most similar code in the training set
            for i in range(len(sim_idx)):
                code_score = code_score_list[i]
                score = code_score
                if score > max_score:
                    max_score = score
                    max_idx = sim_idx[i]
        if self.Is_train==True:   #Training phase: retrieve the most similar code except itself
            code_score_list[code_score_list.index(max(code_score_list))] = 0
            max_idx = sim_idx[code_score_list.index(max(code_score_list))]
        return self.train_code_list[max_idx], self.train_nl_list[max_idx]
