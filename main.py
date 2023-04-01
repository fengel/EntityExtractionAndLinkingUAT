import spacy
import csv
from spacy.kb import KnowledgeBase
import os
from pathlib import Path
import random
from spacy.training import Example
from spacy.ml.models import load_kb
from spacy.util import minibatch, compounding

#load a pretrained English model,
nlp = spacy.load("en_core_web_lg")

#Creating the Knowledge Base
def load_entities():
    entities_loc = Path.cwd() / "data/concepts.csv"  # distributed alongside this notebook
    names = dict()
    descriptions = dict()

    with entities_loc.open("r", encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=";")
        for row in csvreader:
            qid = row[0].replace("http://astrothesaurus.org/uat/","")
            name = row[1]
            desc = ','.join(row).replace(row[0]+","+row[1]+",","",1)
            names[qid] = name
            descriptions[qid] = desc
    return names, descriptions

#Call function to create the dictionary
name_dict, desc_dict = load_entities()

#add each record to the knowledge bas
kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=300)
for qid, desc in desc_dict.items():
    desc_doc = nlp(desc)
    desc_enc = desc_doc.vector
    kb.add_entity(entity=qid, entity_vector=desc_enc, freq=342)   # 342 is an arbitrary value here

#specify aliases or synonyms.
for qid, name in name_dict.items():
    kb.add_alias(alias=name, entities=[qid], probabilities=[1])   # 100% prior probability P(entity|alias)

aliases_loc = Path.cwd() / "data/aliases.csv"

for qid, name in name_dict.items():
    aliases_of_concept = []
    with aliases_loc.open("r", encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=";")
        for row in csvreader:
            if row[0].replace("http://astrothesaurus.org/uat/","") == qid:
                aliases_of_concept.append(row[1])

    if len(aliases_of_concept) != 0:
        for alias in aliases_of_concept:
            kb.add_alias(alias=alias, entities=[qid], probabilities=[1/len(aliases_of_concept)])

# Store knowledge base and nlp model
qids = name_dict.keys()
output_dir = Path.cwd() / "my_output"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
kb.to_disk(output_dir / "my_kb")
nlp.to_disk(output_dir / "my_nlp")

#Creating a training dataset
#Training the Entity Linker
dataset = []
dataset2 = []
file1 = open('data/examples_nocase.txt', 'r')
Lines = file1.readlines()
count = 0

for line in Lines:
    count += 1
    if(len(line.strip().split(":::")) == 6):
        QID = line.strip().split(":::")[1]
        text = line.strip().split(":::")[5]
        offset = (int(line.strip().split(":::")[3]), int(line.strip().split(":::")[4]))
        entity_label = line.strip().split(":::")[0]
        links_dict = {QID: 1.0}
        entities = [(offset[0], offset[1], entity_label)]
        record = (text, {"links": {offset: links_dict}, "entities": entities})
        dataset.append(record)
    elif (len(line.strip().split(":::")) > 6):
        QID = line.strip().split(":::")[1]
        text = line.strip().split(":::")[5]
        entities = []
        links = {}
        for x in range(len(line.strip().split(":::"))//6):
            print(x)
            offset = (int(line.strip().split(":::")[3+(x*6)]), int(line.strip().split(":::")[4+(x*6)]))
            entity_label = line.strip().split(":::")[0]
            links_dict = {QID: 1.0}
            entities.append((offset[0], offset[1], entity_label))
            links.update({offset: links_dict})
            record = (text, {"links": links, "entities": entities})
            if x == (len(line.strip().split(":::"))//6)-1:
                dataset.append(record)
dataset[0]

#check some statistics in this dataset. How many cases of each QID do we have annotated?
gold_ids = []

for text, annot in dataset:
    counted = False
    print(annot["links"].items())
    for span, links_dict in annot["links"].items():

        print(links_dict)
        for link, value in links_dict.items():
            if value:
                counted = True
    if counted:
        gold_ids.append(link)


train_dataset = []
test_dataset = []
count = 0

for QID in qids:
    count = count + 1
    indices = [i for i, j in enumerate(gold_ids) if j == QID]
    last_indice = len(indices)
    if last_indice == 0:
        continue
    print(indices)
    if last_indice > 2:
        training_indice = round(4 * len(indices) / 5)
        test_indice = len(indices) - round(4 * len(indices) / 5)
    elif last_indice == 2:
        training_indice = 1
        test_indice = 1
    else:
        print("testing impossible with a single item dataset.")
        continue

    train_dataset.extend(dataset[index] for index in indices[0:training_indice])  # first X in training
    test_dataset.extend(dataset[index] for index in indices[training_indice:last_indice])  # last Y in test

random.shuffle(train_dataset)
random.shuffle(test_dataset)

# save test data array for external evaluation
with open('my_output/test_dataset', 'w') as outfile:
    outfile.write(repr(test_dataset))
print("done")

TRAIN_EXAMPLES = []
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

sentencizer = nlp.get_pipe("sentencizer")

for text, annotation in train_dataset:
    try:
        example = Example.from_dict(nlp.make_doc(text), annotation)
        example.reference = sentencizer(example.reference)
        TRAIN_EXAMPLES.append(example)
    except:
        print("ambiguity!!!")

#create a new Entity Linking component add it to the pipeline
entity_linker = nlp.add_pipe("entity_linker", config={"incl_prior": False}, last=True)
entity_linker.initialize(get_examples=lambda: TRAIN_EXAMPLES, kb_loader=load_kb(output_dir / "my_kb"))

# train only the entity_linker (disable all other components)
with nlp.select_pipes(enable=["entity_linker"]):
    optimizer = nlp.resume_training()

    for itn in range(500):
        random.shuffle(TRAIN_EXAMPLES)
        batches = minibatch(TRAIN_EXAMPLES, size=compounding(4.0, 32.0, 1.001))  # increasing batch sizes
        losses = {}

        for batch in batches:
            nlp.update(
                batch,
                drop=0.2,      # prevent overfitting
                losses=losses,
                sgd=optimizer,
            )
        if itn % 50 == 0:
            print(itn, "Losses", losses)   # print the training loss

print(itn, "Losses", losses)

#Testing the Entity Linker on unseen data
#Let's first apply it on our original sentence. For each entity, we print the text and label as before, but also the disambiguated QID as predicted by our entity linker.
text = "We have identified a pattern of tectonic deformation on Venus that suggests that many of the planet's lowlands have fragmented into discrete crustal blocks, and that these blocks have moved relative to each other in the geologically recent past."
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_, ent.kb_id_)

#Let's see what the model predicts for the X sentences in our test dataset, that were never seen during training.
for text, true_annot in test_dataset:
    print(text)
    print(f"Gold annotation: {true_annot}")
    doc = nlp(text)  # to make this more efficient, you can use nlp.pipe() just once for all the texts

    for ent in doc.ents:
        #if ent.text == "Mars":
            print(f"Prediction: {ent.text}, {ent.label_}, {ent.kb_id_}")
    print()

# ????? what is done here?
#f = open("data/result.txt", "r")
#count = 0
#identifier = ""
#text = ""

#for t in f:
#    if t.startswith("urn:nasa:"):
#        identifier = t
#    else:
#        print(" ")
#        print(identifier)
#        doc = nlp(t)
#        for ent in doc.ents:
#            print(f"Prediction: {ent.text}, {ent.label_}, {ent.kb_id_}")
