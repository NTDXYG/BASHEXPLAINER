#Obtaining data for the second stage model, data format is  (target code,similarity code,target nl)
from tqdm import tqdm
import pandas as pd
from IR.Faiss_Module import Retrieval
import time

type = "train" # or "valid" or "test"
Is_train = True if type == 'train' else False #Set whether it is a training phase
repo_file = "../data/first_stage_data/train.csv"
target_file = "../data/first_stage_data/"+type+".csv"

df = pd.read_csv(repo_file)
train_code_list = df['code'].tolist()
train_nl_list = df['nl'].tolist()

df = pd.read_csv(target_file)
test_code_list = df['code'].tolist()
test_nl_list = df['nl'].tolist()

def get_sim_info(code):
    sim_code, sim_nl = IR_model.single_query(code, topK=8)
    return {'sim_code':sim_code, 'sim_nl':sim_nl}

if __name__ == "__main__":
    result_list = []
    start_time = time.time()

    IR_model = Retrieval(dim=256, whitening_file='../model/bash_code_vector_whitening.pkl',
                         kernel_file='../model/bash_kernel.pkl',
                         bias_file='../model/bash_bias.pkl', train_code_list=train_code_list,
                         train_nl_list=train_nl_list, Is_train=Is_train)

    print("Sentences to vectors...")
    IR_model.encode_file()

    print("loading index...")
    IR_model.build_index(n_list=1)
    IR_model.index.nprob = 1

    for i in tqdm(range(len(test_code_list))):
        result_list.append(get_sim_info(test_code_list[i])['sim_code'].lower())
    end_time = time.time()
    print(end_time - start_time)
    all_testlist = []
    for i in range(len(result_list)):
        all_testlist.append([test_code_list[i], result_list[i], test_nl_list[i]])
    df = pd.DataFrame(all_testlist, columns=['code', 'similarity', 'nl'])
    df.to_csv("../data/second_stage_data/"+type+".csv", index=False, header=None)  #get retrieval results
