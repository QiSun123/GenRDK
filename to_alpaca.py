import json
import copy
import random

def get_instruction(data_unseen):
    train_instruction = []
    for doc in data_unseen:
        data = copy.deepcopy(doc)
        context = ""
        for sent in doc['sents']:
            context += ' '.join(sent) + " "

        data['context_prompt'] = context

        train_instruction.append(data)

    print("len(train_instruction): ",len(train_instruction))
    return train_instruction


def ToAlpaca_syn(origin_path,output_path,meta_path,rel2id_unseen_path):
    train_unseen=json.load(open(origin_path))
    train_instruction=get_instruction(train_unseen)
    relinfo=json.load(open(meta_path))
    alpaca_unseen=[]

    relation_types = json.load(open(rel2id_unseen_path)).values()
    maxword = 0
    relation_types_str = ", ".join(list(relation_types))
    title_id = 0
    for onedoc in train_instruction:
        item={}
        document = onedoc['context_prompt']
        item["instruction"] =f"""Perform the document-level relation triplet extraction task. Give you the document and pre-defined relation types, you need to extract possible relation triplets. Provide each relation triplet in the following format: (head entity, tail entity, relation type)"""
        item["input"]=f""""The document:\n{document}\nThe pre-defined relation types: "{relation_types_str}"."""
        labels=onedoc['labels']
        output=""
        for label in labels:
            if relinfo[label['r']] in relation_types:
                onelabel="("+onedoc["vertexSet"][label['h']][0]['name']+", "+onedoc["vertexSet"][label['t']][0]['name']+", "+relinfo[label['r']]+")"
                output += onelabel + "\n"
        item["output"]=f"""{output}"""
        if item["output"]=="":
            continue
        item["title"]=onedoc["title"]+ "_" + str(title_id)
        # print(item)
        if len(item["instruction"].split() + item["input"].split() + item["output"].split()) > maxword:
            maxword = len(item["instruction"].split() + item["input"].split() + item["output"].split())
        alpaca_unseen.append(item)
        title_id += 1

    random.shuffle(alpaca_unseen)
    print("len(alpaca_data): ",len(alpaca_unseen))
    print(json.dumps(alpaca_unseen[0],indent=4))
    with open(output_path, 'w') as f1:
        json.dump(alpaca_unseen, f1)
    f1.close()

def ToAlpaca_seen(origin_path,output_path,meta_path,rel2id_seen_path):
    data_seen = json.load(open(origin_path))
    seen_instruction = get_instruction(data_seen)
    relinfo = json.load(open(meta_path))
    alpaca_seen = []
    relation_types = list(json.load(open(rel2id_seen_path)).values())
    way=7
    null=0
    maxword=0
    title_id=0
    for step in range(0,int(len(relation_types)/way)):
        seen_relation_types = relation_types[step*way:step*way+way]
        seen_relation_types_str = ", ".join(seen_relation_types)
        for onedoc in seen_instruction:
            item = {}
            document = onedoc['context_prompt']
            item["instruction"] = f"""Perform the document-level relation triplet extraction task. Give you the document and pre-defined relation types, you need to extract possible relation triplets. Provide each relation triplet in the following format: (head entity, tail entity, relation type)"""
            item["input"] = f"""The document:\n{document}\nThe pre-defined relation types: "{seen_relation_types_str}"."""
            labels = onedoc['labels']
            output = ""
            for label in labels:
                if relinfo[label['r']] in seen_relation_types:
                    onelabel = "(" + onedoc["vertexSet"][label['h']][0]['name'] + ", " + onedoc["vertexSet"][label['t']][0]['name'] + ", " + relinfo[label['r']] + ")"
                    output += onelabel + "\n"
            item["output"] = f"""{output}"""
            item["title"]=onedoc["title"]+"_"+str(title_id)
            item["vertexSet"]=onedoc["vertexSet"]
            item["relation_types"]=seen_relation_types
            if output=="":
                null+=1
            else:
                if len(item["instruction"].split()+item["input"].split()+item["output"].split())>maxword:
                    maxword=len(item["instruction"].split()+item["input"].split()+item["output"].split())
                alpaca_seen.append(item)
                title_id+=1

    random.shuffle(alpaca_seen)
    print("len(alpaca_data): ", len(alpaca_seen))
    print(json.dumps(alpaca_seen[0], indent=4))
    with open(output_path, 'w') as f1:
        json.dump(alpaca_seen, f1)
    f1.close()

ToAlpaca_syn(
    origin_path="synthetic_data_m5_s5/train_synthetic_data.json",
    output_path="alpaca_dataset_m5_s5/alpaca_train_synthetic_data.json",
    meta_path="meta_m5_s5/rel_info.json",
    rel2id_unseen_path="meta_m5_s5/rel2id_unseen.json"
)
ToAlpaca_syn(
    origin_path="synthetic_data_m5_s5/train_denoise_data.json",
    output_path="alpaca_dataset_m5_s5/alpaca_train_denoise_data.json",
    meta_path="meta_m5_s5/rel_info.json",
    rel2id_unseen_path="meta_m5_s5/rel2id_unseen.json"
)

ToAlpaca_seen(
    origin_path="dataset_Redocred_m5_s5/train_seen.json",
    output_path="alpaca_dataset_m5_s5/alpaca_train_seen.json",
    meta_path="meta_m5_s5/rel_info.json",
    rel2id_seen_path="meta_m5_s5/seenrel2id_info.json"
)

ToAlpaca_seen(
    origin_path="dataset_Redocred_m5_s5/dev_seen.json",
    output_path="alpaca_dataset_m5_s5/alpaca_dev_seen.json",
    meta_path="meta_m5_s5/rel_info.json",
    rel2id_seen_path="meta_m5_s5/seenrel2id_info.json"
)

ToAlpaca_seen(
    origin_path="dataset_Redocred_m5_s5/test_seen.json",
    output_path="alpaca_dataset_m5_s5/alpaca_test_seen.json",
    meta_path="meta_m5_s5/rel_info.json",
    rel2id_seen_path="meta_m5_s5/seenrel2id_info.json"
)


# ToAlpaca_seen(group,name="train_seen.json")
# ToAlpaca_seen(group,name="dev_seen.json")
# ToAlpaca_seen(group,name="test_seen.json")







