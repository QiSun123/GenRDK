import json
import os

import numpy as np
import re

from tqdm import tqdm

def findMention(sents, ent):
    mentions = []
    sent_id = 0
    for sent in sents:
        i = 0
        lowersent = [s.lower() for s in sent]
        for word in lowersent:
            if word == ent[0].lower():
                lowerent = [e.lower() for e in ent]
                if lowerent == lowersent[i:(i + len(ent))]:
                    mention = {}
                    mention['name'] = ' '.join(sent[i:(i + len(ent))])
                    mention['sent_id'] = sent_id
                    mention['pos'] = [i, i + len(ent)]
                    mentions.append(mention)

            i += 1
        sent_id += 1
    return mentions

def denoise(predictResult,rel2id_unseen,train_unseen,savePath):
    result = json.load(open(predictResult))
    mention_predict = []
    predict_graph = {}
    relation_code = json.load(open(rel2id_unseen))
    relation_list = list(relation_code.keys())
    # Construct the knoweledge graph from pseudo label
    for item in result:
        if item['r'] in relation_list:
            if item['h_idx'] not in mention_predict:
                mention_predict.append(item['h_idx'])
            if item['t_idx'] not in mention_predict:
                mention_predict.append(item['t_idx'])
            triplet = (item['h_idx'], item['t_idx'], item['r'])
            if triplet not in predict_graph.keys():
                predict_graph[triplet] = 0
            predict_graph[triplet] += 1

    train_unseen = json.load(open(train_unseen))

    # Construct the knoweledge graph from synthetic data
    gpt_graph = {}
    for item in train_unseen:
        mention_list = []
        for ent in item['vertexSet']:
            mention_list.append(ent[0]['name'])
        for label in item['labels']:
            if label['r'] in relation_list:
                triplet = (mention_list[label['h']].lower(), mention_list[label['t']].lower(), label['r'])
                if triplet not in gpt_graph.keys():
                    gpt_graph[triplet] = 0
                gpt_graph[triplet] += 1
    #Fuse the graph
    merge_graph = {}
    same_triplet = 0
    for triplet in gpt_graph:
        if triplet in predict_graph.keys():
            same_triplet += 1
            merge_graph[triplet] = gpt_graph[triplet] + predict_graph[triplet]
        else:
            merge_graph[triplet] = gpt_graph[triplet]
    for triplet in predict_graph:
        if triplet not in merge_graph.keys():
            merge_graph[triplet] = predict_graph[triplet]
    statis_graph = {}
    th_fused = {}
    # Calculate the consistency score
    for type in relation_list:
        statis_graph[type] = []
        th_fused[type] = 0

    for triplet in merge_graph:
        statis_graph[triplet[2]].append(merge_graph[triplet])
    # Set the thresholds
    for type in statis_graph:
        print(statis_graph[type])
        static_div=sorted(statis_graph[type])[int(0.10 * len(statis_graph[type])):int(0.8 * len(statis_graph[type]))] # optional parameter
        temp_th_fused= np.mean(static_div)-np.std(static_div, ddof=1)
        if temp_th_fused>0:
            th_fused[type] = temp_th_fused
        else:
            th_fused[type] = 0
    # Denoise the graph with consistency score and thresholds
    deoise_graph = {}
    for triplet in merge_graph:
        if triplet in predict_graph.keys() and triplet in gpt_graph.keys():
            deoise_graph[triplet] = merge_graph[triplet]
        elif merge_graph[triplet] > th_fused[triplet[2]]:
            deoise_graph[triplet] = merge_graph[triplet]
    print("deoise_graph:", len(deoise_graph))
    print("predict_graph:", len(predict_graph))
    print("gpt_graph:", len(gpt_graph))

    # Relabel and filter the synthetic data
    train_denoise = []
    filter=0
    for i in tqdm(range(0,len(train_unseen))):
        item=train_unseen[i]
        one_denoise = {}
        one_denoise['sents'] = item['sents']
        one_denoise['title'] = item['title']
        vetexs = []
        labels = []
        entity_id_map = {}
        trip_check= []
        for triplet in deoise_graph:
            head = re.findall(r'\w+|[^\w\s]', triplet[0])
            tail = re.findall(r'\w+|[^\w\s]', triplet[1])
            rel = triplet[2]
            entity_h = findMention(item["sents"], head)
            entity_t = findMention(item["sents"], tail)
            if entity_t != [] and entity_h != []:
                # print(triplet)
                if entity_h[0]['name'] not in entity_id_map.keys():
                    entity_id_map[entity_h[0]['name']] = len(vetexs)
                    vetexs.append(entity_h)
                if entity_t[0]['name'] not in entity_id_map.keys():
                    entity_id_map[entity_t[0]['name']] = len(vetexs)
                    vetexs.append(entity_t)
                if (entity_id_map[entity_h[0]['name']], entity_id_map[entity_t[0]['name']], rel) not in trip_check:
                    trip_check.append((entity_id_map[entity_h[0]['name']], entity_id_map[entity_t[0]['name']], rel))
                    labels.append({'h': entity_id_map[entity_h[0]['name']], 't': entity_id_map[entity_t[0]['name']], 'r': rel})

        one_denoise['vertexSet'] = vetexs
        if len(labels) == 0:
            filter += 1
        else:
            one_denoise['labels'] = labels
            train_denoise.append(one_denoise)

    print("train_denoise:", len(train_denoise))
    print("filter:", filter)
    with open(savePath, 'w') as f:
        json.dump(train_denoise, f)
    print("save in: ", savePath)
    print("finish!")

denoise(predictResult="", # put your path of pseudo labels here
        rel2id_unseen="meta_m5_s5/rel2id_unseen.json",
        train_unseen="synthetic_data_m5_s5/train_synthetic_data.json",
        savePath="synthetic_data_m5_s5/train_denoise_data.json")

'''
The format of predictResult is like:
[
    {"title": "The Godfather id_522", "h_idx": "The Godfather", "t_idx": "Francis Ford Coppola", "r": "P58"},
] 
'''