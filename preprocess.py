import random
from bashlint.data_tools import *
from nlp_tools import tokenizer
import pandas as pd
import tqdm
import shlex

def clean(text):
    if(text.startswith('(')):
        text = text[text.find(')')+2:]
    return text.lower()

# load data from NLC2CMD competition
cmd_df = pd.read_csv("data/raw_data/all.cm.filtered", header=None, delimiter='\n')
cmd_list = cmd_df[0].tolist()
nl_df = pd.read_csv("data/raw_data/all.nl.filtered", header=None, delimiter='\n')
nl_list = nl_df[0].tolist()

# load data from nl2bash paper
df = pd.read_json("data/raw_data/nl2bash-data.json", orient='index')
nl_list.extend(df['invocation'].tolist())
nl_list = [clean(t) for t in nl_list]
cmd_list.extend(df['cmd'].tolist())

# merge these two dataset
temp_list = []
data_list = []
for i in tqdm.tqdm(range(len(nl_list))):
    if cmd_list[i] not in temp_list:
        data_list.append([cmd_list[i], nl_list[i]])
    temp_list.append(cmd_list[i])
    temp_list.append(nl_list[i])
    temp_list = list(set(temp_list))


# parse and preprocess
def preprocess(data_list):
    cmd_list, nl_list = [], []
    result_list = []
    for i in range(len(data_list)):
        cmd_list.append(data_list[i][0])
        nl_list.append(data_list[i][1])
    for i in tqdm.tqdm(range(len(cmd_list))):
        try:
            cmd = ' '.join(bash_tokenizer(cmd_list[i], loose_constraints=True, arg_type_only=True))
            if (cmd == ""):
                cmd = " ".join(shlex.split(cmd_list[i]))
            nl = ' '.join(tokenizer.ner_tokenizer(nl_list[i])[0])
            result_list.append([cmd, nl])
        except:
            print("fail")
            continue

    return result_list

data_list = preprocess(data_list)

# random shuffle
random.shuffle(data_list)

length = int(len(data_list)/10)
train_list = data_list[:8*length]
valid_list = data_list[8*length:9*length]
test_list = data_list[9*length:]

df = pd.DataFrame(train_list, columns=['code', 'nl'])
df.to_csv("data/first_stage_data/train.csv", index=False)

valid_list = preprocess(valid_list)
df = pd.DataFrame(valid_list, columns=['code', 'nl'])
df.to_csv("data/first_stage_data/valid.csv", index=False)

test_list = preprocess(test_list)
df = pd.DataFrame(test_list, columns=['code', 'nl'])
df.to_csv("data/first_stage_data/test.csv", index=False)