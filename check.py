import json
unseen_types=json.load(open("meta_m5_s5/rel2id_unseen.json")).keys()
seen_train=json.load(open("dataset_Redocred_m5_s5/train_seen.json"))
seen_dev=json.load(open("dataset_Redocred_m5_s5/dev_seen.json"))
seen_test=json.load(open("dataset_Redocred_m5_s5/test_seen.json"))
seen_data=seen_train+seen_dev+seen_test
flag=0
for doc in seen_data:
    labels=doc['labels']
    for label in labels:
        if label["r"] in unseen_types:
            print("Error: Unseen relation in seen data")
            print(label)
            flag=1
if flag==0:
    print("Check passed")
else:
    print("Check failed")

