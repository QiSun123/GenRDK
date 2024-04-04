import copy
import json
import random
import re
import openai
import time
import nltk
from nltk.tokenize import sent_tokenize
import tqdm

# put your openai api key here

openai.api_key = "private openai api"
save_history_dir='synthetic_data_m5_s5/history/'
metadir='meta_m5_s5/'
savedor='synthetic_data_m5_s5/'
relation_prompt=metadir+'relation_prompt.json'
unseen_rel2id=metadir+'unseenrel2id.json'

print("openai.api_key: ",openai.api_key)
print("relation_prompt: ",relation_prompt)
print("save_history_dir: ",save_history_dir)

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message["content"]

def generate(relation_types):
    # key_relation is the target unseen relation type, relationStr is the related relation types (include target relation type) selected by chatgpt
    for key_relation in relation_types:
        print("key: ",key_relation)
        relationStr="\", \"".join(relation_types[key_relation])
        print(relationStr)
        # all_relations are the relation types in ReDocRED
        all_relations="head of government', 'country', 'place of birth', 'place of death', 'father', 'mother', 'spouse', 'country of citizenship', 'continent', 'instance of', 'head of state', 'capital', 'official language', 'position held', 'child', 'author', 'member of sports team', 'director', 'screenwriter', 'educated at', 'composer', 'member of political party', 'employer', 'founded by', 'league', 'publisher', 'owned by', 'located in the administrative territorial entity', 'operator', 'religion', 'contains administrative territorial entity', 'follows', 'followed by', 'headquarters location', 'cast member', 'producer', 'award received', 'creator', 'parent taxon', 'ethnic group', 'performer', 'manufacturer', 'developer', 'series', 'sister city', 'legislative body', 'basin country', 'located in or next to body of water', 'military branch', 'record label', 'production company', 'location', 'subclass of', 'subsidiary', 'part of', 'original language of work', 'platform', 'mouth of the watercourse', 'original network', 'member of', 'chairperson', 'country of origin', 'has part', 'residence', 'date of birth', 'date of death', 'inception', 'dissolved, abolished or demolished', 'publication date', 'start time', 'end time', 'point in time', 'conflict', 'characters', 'lyrics by', 'located on terrain feature', 'participant', 'influenced by', 'location of formation', 'parent organization', 'notable work', 'separated from', 'narrative location', 'work location', 'applies to jurisdiction', 'product or material produced', 'unemployment rate', 'territory claimed by', 'participant of', 'replaces', 'replaced by', 'capital of', 'languages spoken, written or signed', 'present in work', 'sibling"
        all_relations=all_relations.replace("', '","\", \" ")
        # optional, you can change the content_theme to any themes
        content_theme="a famous individual, military force, melodious song, renowned club, political party, captivating book, opera, video game, film, TV serie, remarkable country, mesmerizing artwork, distinguished organization or influential company in the world"
        messages = [
            {'role': 'system', 'content': 'You are an information extraction assistant.'}]
        prompts = [
            f"""Generate one fictional wikipedia style paragraph that contains at least 6 sentences and describes one or more following relation types: "{relationStr}”. The content of this generated fictional wikipedia style paragraph can be the description of {content_theme}. Provide them in JSON format with just following keys: title, context.""",
            f"""Extract the entities in your above generated document. Provide them in List of JSON format with the following keys: entity, entity type. The entity type can be one of the following types: "Organization", "Location", "Time", "Person", "Miscellaneous", "Number", "Blank". """,
            f"""Present the relational triplets as (one head entity, one tail entity, one relation type), if a relationship exists between two entities. The relation type can be one or more of following relation types: "{all_relations}”.""",
            f"""Present reasoning explannation of each relational triplet.""",
            f"""Present support sentence index for each extracted relational triple that shown in your generated document.""",
            f"""Organize the above triplet information in List of JSON format with the following keys: head entity, tail entity, relation type, reasoning explannation of each relational triplet, index of supporting sentence that shown in document,""",
        ]
        index = 0
        for pro in prompts:
            messages.append({'role': 'user', 'content': f"""{pro}"""})
            flag = 1
            thref = 0
            while flag:
                try:
                    if index == 0:
                        temperature = 1
                    else:
                        temperature = 0.0
                    response = get_completion_from_messages(messages, temperature=temperature)

                except Exception as e:
                    print(e)
                    if thref >= 3:
                        break
                    print("wait 2 seconds!")
                    thref += 1
                    time.sleep(2)
                else:
                    print("success!")
                    flag = 0
            if flag == 1:
                # one of the prompts is not generated
                break
            messages.append({'role': 'assistant', 'content': f"""{response}"""})
            index += 1
        if index != len(prompts):
            # generation is not successful
            print("skip this doc!")
            continue
        save_path = save_history_dir + 'history_' + key_relation + '.json'
        try:
            history = json.load(open(save_path))
        except Exception as e:
            history = []
        history.append(messages)
        history = json.dumps(history)
        with open(save_path, 'w') as outfile:
            outfile.write(history)
        outfile.close()

def static():
    history=[]
    relation_types = json.load(open(relation_prompt))
    for key_relation in relation_types:
        try:
            load_path = save_history_dir +  '/history_' + key_relation + '.json'
            his = json.load(open(load_path))
        except Exception as e:
            print("no such file!")
            print(load_path)
            his = []
        for hi in range(0, len(his)):
            item = his[hi]
            temp = {'role': 'tag'}
            temp['content'] = key_relation
            item.append(temp)
            history.append(item)

    print("Number of origin synthetic data in current seed: ",len(history))
    dataset=[]
    index=0
    for message in history:
        print("index: ", index)
        onedata={}
        try:
            doc=message[2]['content']
            doc=doc.replace("\n","")
            doc=doc.replace("\"Hakuna Matata.\"", "\\\"Hakuna Matata\\\"")
            doc = doc.replace("Here is a fictional paragraph that describes the relation types \"place of birth\", \"place of death\", \"father\", \"mother\", and \"position held\":", "")
            doc = doc.replace("Here's an example paragraph:", "")
            doc = doc.replace("```{", "{")
            doc = doc.replace("```This paragraph contains relations \"place of birth\" (Pella), \"father\" (King Philip II of Macedonia), \"mother\" (Queen Olympia), \"position held\" (King of Macedonia), \"place of death\" (Babylon).", "")
            doc = doc.replace("JSON Format:", "")
            doc=doc.replace("Relation Types:- Place of birth: \"Stagira, Greece\"- Father: \"Nicomachus\"- Mother: \"Phaestis\"- Place of death: \"Euboea, Greece\"- Position held: \"Philosopher and scientist\"", "")
            doc=doc.replace("As an information extraction assistant, I have generated a fictional paragraph with the following relation types: \"place of birth,\" \"place of death,\" \"father,\" \"mother,\" and \"position held.","")
            doc=doc.replace("\"New Beginnings,\"","\\\"New Beginnings\\\",")
            doc = doc.replace("\"In Memory Of,\"", "\\\"In Memory Of\\\",")
            doc = doc.replace("\"The Lost Souls\"", "\\\"The Lost Souls\\\"")
            doc=doc.replace("\"{  \"title\"","{  \"title\"")
            doc=doc.replace("\"Shallow\"","\\\"Shallow\\\"")
            doc=doc.replace("```json{","{")
            doc=doc.replace("JSON format:{","{")
            doc=json.loads(doc)
        except Exception as e:
            print("*********** without content ***********")
            index += 1
            continue
        try:
            onedata['title']=doc['title']
            onedata['sents']=doc['context']
        except Exception as e:
            print("*********** without title or content ***********")
            index += 1
            continue

        try:
            entities=message[4]['content']
            entities=entities.replace("\n","")
            entities = entities.replace("},]", "}]")
            entities = entities.replace("\"}    {\"", "}, {")
            entities = entities.replace("{entity", "{\"entity")
            entities = entities.replace("\"Miscellaneous}", "\"Miscellaneous\"}")
            entities = entities.replace("JSON Format:", "")
            entities = entities.replace("```[", "[")
            entities = entities.replace("```The identified entity types are:- Person: Alexander the Great, Philip II of Macedonia, Queen Olympia, Aristotle- Location: Pella, Macedonia, Persian Empire, Hellenistic, Babylon- Time: July 356 BC, June 10, 323 BC- Number: 20, 32- Miscellaneous: Hellenistic, Greek", "")
            entities = entities.replace("Here are the extracted entities from the document:", "")
            entities=entities.replace("Here are the extracted entities in List of JSON format with the following keys: entity, entity type:","")
            entities=entities.replace("Note: The entity type for \"384 BC\" is \"Time\" and \"Philosopher and scientist\" is \"Miscellaneous\".","")
            entities=json.loads(entities)

        except Exception as e:

            print("*********** entity ***********")

            index += 1
            continue
        vertexSet=[]
        for entity in entities:
            vertex={}
            try:
                vertex['name']=entity['entity']
                if 'entity type' in entity.keys():
                    vertex['type']=entity['entity type']
                elif 'entity_type' in entity.keys():
                    vertex['type']=entity['entity_type']
                else:
                    print("*********** vertex keys ***********")

            except Exception as e:
                print("*********** vertex part ***********")

                index += 1
                continue
            vertexSet.append(vertex)
        onedata['vertexSet']=vertexSet

        try:
            relations=message[12]['content']
            relations = relations.replace("\n", "")
            relations=relations.replace("Here's the triplet information organized in List of JSON format, as per your request:```","")
            relations = relations.replace("```Note: The remaining seven triples cannot be supported or explained by the provided context or they might be incorrect interpretations.", "")
            relations = relations.replace("\": None","\": \" \" ")
            relations = relations.replace("Sure, here it is::", "")
            relations = relations.replace("Here is the information organized in a JSON format with the requested keys for each relational triplet:```", "")
            relations = relations.replace("]```", "}]")
            relations = relations.replace("JSON Format:","")
            relations = relations.replace("```[", "[")
            relations= relations.replace("```For each triplet, the JSON object includes:- head entity: the main entity in the relationship- tail entity: the secondary entity in the relationship- relation type: the type of relationship between the entities- reasoning: an explanation of the significance of the relationship- context: the relevant sentence in which the relationship is mentioned in the original document.", "")
            relations = relations.replace("Sure, here is the information organized in a list of JSON format:```", "")
            relations = relations.replace("Sure, here it is:", "")
            relations=relations.replace("}}]","}]")
            relations=relations.replace("}}]For each triplet, the JSON object includes:- head entity: the main entity in the relationship- tail entity: the secondary entity in the relationship- relation type: the type of relationship between the entities- reasoning: an explanation of the significance of the relationship- context: the relevant sentence in which the relationship is mentioned in the original document.","}]")
            relations=relations.replace("Sure, here is the information organized in a list of JSON format:","")
            relations=relations.replace("Sure, here is the information organized in a list of JSON format with the requested keys:","")
            relations=relations.replace("Sure, here is the organized triplet information in List of JSON format:","")
            relations=relations.replace("This JSON format organizes the extracted relational triplets, along with their reasoning explanations and supporting sentences, in a structured and easy-to-read format.","")
            relations=relations.replace("Sure, here is the information organized in a list of JSON format with the requested keys:","")
            relations=relations.replace("I apologize for the confusion earlier, but I did not generate any relational triplets in the previous document. However, I can generate a list of JSON format for the entities extracted from the document with their corresponding support sentences. Here is the list:","")
            relations=relations.replace("I apologize for the confusion earlier. Here are the relational triplets based on the generated document, organized in a list of JSON format with the requested keys:","")
            relations=relations.replace("I apologize for the confusion earlier, as the document I generated did not contain any personal relationships or positions held by individuals. Therefore, I cannot provide reasoning explanations or supporting sentences for the extracted relational triplets.However, I can provide a general explanation of how the information can be organized in a list of JSON format. The list can contain one or more relational triplets, each represented as a JSON object with the following keys:- \"head entity\": the entity that appears as the subject of the relationship- \"tail entity\": the entity that appears as the object of the relationship- \"relation type\": the type of relationship between the head and tail entities- \"reasoning explanation\": a brief explanation of how the relationship was inferred from the text- \"supporting sentence\": the complete context of the sentence in the text that provides evidence for the relationshipHere is an example of how the information can be organized in a list of JSON format:","")
            relations=relations.replace("Sure, here is the organized information in List of JSON format:","")
            relations=relations.replace("Sure, here is the organized triplet information in a list of JSON format:","")
            relations=relations.replace("As mentioned earlier, I cannot generate relational triplets for the above document as it does not contain any information related to personal relationships or positions held by individuals. However, if there were such information, I could organize the triplet information in a list of JSON format with the following keys:","")
            relations=relations.replace("Note that this is just an example, and the actual JSON format may vary depending on the specific information and relationships extracted from the text.","")
            relations=relations.replace("Sure, here is the information organized in List of JSON format","")
            relations=relations.replace("I apologize for the confusion earlier. As mentioned earlier, there are no relational triplets in the generated document. However, I can provide a list of JSON format for each extracted entity in the document with the following keys:- Head entity- Tail entity- Relation type- Reasoning explanation of each relational triplet- Complete context of supporting sentence that shown in document","")
            relations=relations.replace("As there are no relational triplets in the generated document, I cannot provide the requested information. However, if there were relational triplets, the JSON format with the requested keys would look like this:","")
            relations=relations.replace("cinematic experience.\"}","cinematic experience.\"}]")
            relations=relations.replace("Since there were no relational triplets in the generated document, I will provide a list of JSON format for each extracted entity with the following keys: entity, entity type, and supporting sentence. Here is the list:","")
            relations=relations.replace("Sure, here is the list of JSON format with the requested keys:","")
            relations=relations.replace("As there are no relational triplets in the generated document, I cannot provide the requested information. However, I can provide a sample JSON format for a relational triplet:```","")
            relations=relations.replace("Sure, here's the information organized in a list of JSON format:","")
            relations=relations.replace("For each triplet, the JSON object includes:- head entity: the main entity in the relationship- tail entity: the secondary entity in the relationship- relation type: the type of relationship between the entities- reasoning: an explanation of the significance of the relationship- context: the relevant sentence in which the relationship is mentioned in the original document.","")
            relations=relations.replace(":[    {","[{")
            relations=relations.replace(":[  {","[{")
            relations=relations.replace("```json{","{")

            relations=json.loads(relations)

        except Exception as e:
            print("*********** relations part ***********")
            index += 1
            continue
        labels=[]
        if len(relations)<=1:
            print("*********** structure check ***********")

        for rela in relations:
            onerelation={}
            try:
                if "head_entity" in rela.keys():
                    onerelation['h']=rela["head_entity"]
                elif "head entity" in rela.keys():
                    onerelation['h']=rela["head entity"]
                else:
                    print("*********** h labels part ***********")

                    continue
                if "tail_entity" in rela.keys():
                    onerelation['t']=rela["tail_entity"]
                elif "tail entity" in rela.keys():
                    onerelation['t']=rela["tail entity"]
                else:
                    print("*********** t labels part ***********")

                    continue

                if "relation_type" in rela.keys():
                    onerelation['r']=rela["relation_type"]
                elif "relation type" in rela.keys():
                    onerelation['r']=rela["relation type"]
                else:
                    print("*********** r labels part ***********")

                    continue

                if "reasoning_explanation" in rela.keys():
                    onerelation['reasoning']=rela["reasoning_explanation"]
                elif "reasoning explanation" in rela.keys():
                    onerelation['reasoning']=rela["reasoning explanation"]
                elif "explanation" in rela.keys():
                    onerelation['reasoning']=rela["explanation"]
                elif "reasoning explannation" in rela.keys():
                    onerelation['reasoning']=rela["reasoning explannation"]
                elif "reasoning" in rela.keys():
                    onerelation['reasoning']=rela["reasoning"]
                elif "reasoning explanantion" in rela.keys():
                    onerelation['reasoning']=rela["reasoning explanantion"]

                else:
                    print("*********** reasoning_explanation labels part ***********")
                    onerelation['reasoning']=""

                if "supporting_sentence" in rela.keys():
                    onerelation['evidence']=rela["supporting_sentence"]
                elif "supporting sentence" in rela.keys():
                    onerelation['evidence']=rela["supporting sentence"]
                elif "complete context" in rela.keys():
                    onerelation['evidence']=rela["complete context"]
                elif "supporting sentence context" in rela.keys():
                    onerelation['evidence']=rela["supporting sentence context"]
                elif "complete context of supporting sentence" in rela.keys():
                    onerelation['evidence']=rela["complete context of supporting sentence"]
                elif "context" in rela.keys():
                    onerelation['evidence']=rela["context"]
                elif "complete context sentence" in rela.keys():
                    onerelation['evidence']=rela["complete context sentence"]
                elif "supporting context" in rela.keys():
                    onerelation['evidence']=rela["supporting context"]
                elif "supporting_context" in rela.keys():
                    onerelation['evidence']=rela["supporting_context"]
                elif "complete_supporting_sentence" in rela.keys():
                    onerelation['evidence']=rela["complete_supporting_sentence"]
                else:
                    print("*********** evidence labels part ***********")
                    onerelation['evidence']=""
            except Exception as e:
                print("*********** labels part ***********")
                continue
            if onerelation['t']!=None and onerelation['r']!=None and onerelation['h']!=None:
                labels.append(onerelation)
            else:
                print(" *********** None part ***********")

        if len(labels)!=0:
            onedata['labels']=labels
        else:
            print(" *********** labels is null ***********")
            continue
        onedata["relation_tag"]=message[13]['content']
        dataset.append(onedata)
        index+=1
    print(len(dataset))
    print(len(history))
    print("saving rate: " ,(len(dataset)/len(history))*100)
    filterpath=savedor+'FilterDataset.json'
    dataset=json.dumps(dataset)
    with open(filterpath,'w') as f1:
        f1.write(dataset)
    f1.close()
    print("filter path: ",filterpath)

def findMention(sents,ent,type):
    mentions = []
    sent_id = 0
    for sent in sents:
        i = 0
        lowersent=[s.lower() for s in sent]
        for word in lowersent:
            if word==ent[0].lower():
                lowerent=[e.lower() for e in ent]
                if lowerent==lowersent[i:(i+len(ent))]:
                    mention = {}
                    mention['name'] = ' '.join(sent[i:(i+len(ent))])
                    mention['sent_id'] = sent_id
                    mention['type'] = type
                    mention['pos']=[i,i+len(ent)]
                    mentions.append(mention)

            i+=1
        sent_id+=1
    return mentions

def findSentence(sents,name):
    evidences=[]
    sent_id=0
    for sent in sents:
        lowesent=[s.lower() for s in sent]
        lowername=[n.lower() for n in name]
        if lowername==lowesent:
            evidences.append(sent_id)
        sent_id+=1
    return evidences

def inverse(label_relation):
    inverse={
        "author": "notable work",
        "performer":"notable work",
        "producer":"notable work",
        "composer":"notable work",
        "director":"notable work",
        "lyrics by":"notable work",
        "participant":"participant of",
        "participant of":"participant",
        "has part":"part of",
        "sibling":"sibling",
        "series":"has part",
        "spouse":"spouse",
        "characters":"present in work",
        "conflict":"participant",
        "parent organization":"subsidiary",
        "subsidiary":"parent organization",
        "follows":"followed by",
        "followed by":"follows",
        "father":"child",
        "replaced by":"replaces",
        "head of government":"applies to jurisdiction",
        "replaces":"replaced by",
        "legislative body":"applies to jurisdiction",
        "head of state":"applies to jurisdiction",
        "mother":"child",
        "part of":"has part",
        "sister city":"sister city",
        "capital":"capital of"
    }
    relinfo=json.load(open(metadir+'rel_info.json'))
    info2rel={}
    for key in relinfo:
        info2rel[relinfo[key]]=key
    inverseid={}
    for key in inverse:
        inverseid[info2rel[key]]=info2rel[inverse[key]]
    htr=[]
    for item in label_relation:
        htr.append(str(item['h'])+"_"+str(item['t'])+"_"+item['r'])
    for item in label_relation:
        if item['r'] in inverseid.keys():
            if str(item['t'])+"_"+str(item['h'])+"_"+inverseid[item['r']] not in htr:
                label_relation.append({'h':item['t'],'t':item['h'],'r':inverseid[item['r']], 'evidence':item['evidence'],'reasoning':item['reasoning']})
                htr.append(str(item['t']) + "_" + str(item['h']) + "_" + inverseid[item['r']])
    return label_relation

def transfer():
    origin=json.load(open(savedor+'FilterDataset.json'))
    datasets=[]
    nltk.download('punkt')
    index=0
    typeMap = {"Blank": "BLANK", "Organization": "ORG", "Location": "LOC", "Time": "TIME", "Person": "PER",
               "Miscellaneous": "MISC", "Number": "NUM"}
    rel_info=json.load(open(metadir+'rel_info.json'))
    info2rel={}
    error5=0
    relationfact=0
    entityNumber=0
    for key in rel_info:
        info2rel[rel_info[key]]=key

    SameContext=[]

    for doc in origin:
        data={}
        context=doc['sents']
        data['title'] = doc['title']+" id_"+str(len(datasets))
        try:
            context=sent_tokenize(context)

        except Exception as e:
            print(" *********** Filter by error 1 ***********")

            index+=1
            continue

        if ' '.join(context) in SameContext:
            print(" *********** Same context 1.1 *********** ")

        else:
            SameContext.append(' '.join(context))
        sentences=[]
        for sent in context:
            sentlist=re.findall(r'\w+|[^\w\s]', sent)
            sentences.append(sentlist)
        data['sents']=sentences

        entityL=[]
        for entity in doc["vertexSet"]:
            entityL.append(entity['name'])
        entities=[]
        entityR=[]
        for relaE in doc['labels']:

            if type(relaE['h'])==str and relaE['h'] not in entityR:
                entityR.append(relaE['h'])
            if type(relaE['t'])==str and relaE['t'] not in entityR:
                entityR.append(relaE['t'])

            if relaE['h'] not in entityL and relaE['h'] !='' and type(relaE['h'])==str:
                entities.append({'name':relaE['h'],'type':"Miscellaneous"})
            if relaE['t'] not in entityL and relaE['t'] !='' and type(relaE['t'])==str:
                entities.append({'name':relaE['t'],'type':"Miscellaneous"})
        for relaE in doc["vertexSet"]:
            if relaE['name'] in entityR:
                entities.append(relaE)

        vertexSet=[]
        entity_idmap={}
        for entity in entities:

            name=re.findall(r'\w+|[^\w\s]', entity['name'])
            try:
                mentionType=typeMap[entity['type']]
            except Exception as e:

                continue

            vertex = findMention(sentences,name,mentionType)
            if len(vertex)!=0:
                vertexSet.append(vertex)
                entity_idmap[' '.join(name)] = len(vertexSet) - 1
            else:
                print(" *********** can not find mention in doc ***********")

        data['vertexSet']=vertexSet
        relations=doc['labels']
        label_relation=[]
        for relation in relations:
            relaItem={}
            try:
                relaName=relation['r']
                if type(relaName)==list:

                    for overlap_r in relaName:
                        overlap_relation=copy.deepcopy(relation)
                        overlap_relation['r']=overlap_r
                        relations.append(overlap_relation)
                    continue

                if relaName=="located in":
                    relaName="located in the administrative territorial entity"
                elif relaName=="directed":
                    relaName="director"
                elif relaName=="directing and co-writing":
                    relaName = "director"
                if relaName=="birthplace":
                    relaName="place of birth"
                if relaName=="famous work":
                    relaName="notable work"
                if relaName=="birth location":
                    relaName="place of birth"
                if relaName=="occupation":
                    relaName="position held"
                if relaName=="Occupation":
                    relaName="position held"
                if relaName=="date of publication":
                    relaName="publication date"

                relaItem['r'] = info2rel[relaName.lower()]
            except Exception as e:

                error5 += 1
                continue

            try:
                if type(relation['h'])==list:
                    head=' '.join(relation['h'])
                    tail=' '.join(relation['t'])
                else:
                    head=relation['h']
                    tail=relation['t']
                head=' '.join(re.findall(r'\w+|[^\w\s]', head))
                tail=' '.join(re.findall(r'\w+|[^\w\s]', tail))
                relaItem['h'] = entity_idmap[head]
                relaItem['t'] = entity_idmap[tail]

            except Exception as e:

                error5 += 1

                continue
            if relaItem['h']==relaItem['t']:

                continue


            if relation['evidence']==[] or relation['evidence']=='':

                relaItem['evidence']=[]
            else:

                if type(relation['evidence'])==str:
                    evidence=re.findall(r'\w+|[^\w\s]', relation['evidence'])
                    relaItem['evidence'] = findSentence(sentences, evidence)
                else:
                    evilist=[]
                    for onevi in evidence:
                        onevi=re.findall(r'\w+|[^\w\s]', onevi)
                        evilist+=findSentence(sentences, onevi)
                    relaItem['evidence'] = evilist

            relaItem['reasoning'] = relation['reasoning']
            label_relation.append(relaItem)

        if len(label_relation)==0:
            print(" *********** No relational fact in this Doc 6 ***********")
            index += 1
            continue
        label_relation=inverse(label_relation)
        relationfact+=len(label_relation)
        data['labels']=label_relation
        entityNumber+=len(data['vertexSet'])
        unseen_reltype=list(json.load(open(metadir+'rel2id_unseen.json')).keys())
        keyRcount=0
        for theR in data['labels']:
            if theR['r'] in unseen_reltype:
                keyRcount+=1
        if keyRcount<0:
            print(" *********** No target relation in this Doc 7_1 ***********")
            index += 1
            continue

        if len(data['labels'])<6 :
            print(" *********** No enough relational fact in this Doc 7 ***********")
            index += 1
            continue

        data['tag']=doc['relation_tag']
        datasets.append(data)
        index+=1

    dis_generate = {}
    relation_types = json.load(open(relation_prompt))

    for key in relation_types.keys():
        dis_generate[info2rel[key]] = 0

    for item in datasets:
        labels = item['labels']
        for labe in labels:
            if labe['r'] in dis_generate.keys():
                dis_generate[labe['r']] += 1


    print("saving rate: ", (len(datasets) / len(origin)) * 100)
    print("***************** statistic ******************")
    print("len of datasets: ", len(datasets))
    print("avg relational fact: ", (relationfact / len(datasets)) )
    print("avg entity: ", (entityNumber / len(datasets)))
    print("distribution: ", dis_generate)

    datasets=json.dumps(datasets)
    unseenpath=savedor+'train_synthetic_data.json'
    with open(unseenpath, 'w') as f:
        f.write(datasets)
    f.close()
    print("save path: ", unseenpath)

def main():
    num=1
    # relation_types = json.load(open(relation_prompt))
    # for i in range(num):
    #     print("***********",i,"***********")
    #     generate(relation_types)
    # static()
    print("transfering...")
    transfer()

main()




