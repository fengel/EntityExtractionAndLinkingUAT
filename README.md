# EntityExtractionAndLinkingUAT
Pre requisites:

  !pip install spacy==2.2.4 

  !python -m spacy download en_core_web_lg

1) Create Training data:

 train_costume_ner_model.py: takes as input some training data, converts it to a DocBin Object
and stores that Object as "train.spacy"

2) Create model:

The file config.cfg holds the configuration for the training process. The training that bases on that configuration and the "train.spacy" model is called via CLI: python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./train.spacy

3) Evaluate Model

The python code "evaluate_existing_models.py" loads the trained model from step 2) does a first simple check and next calculates measures