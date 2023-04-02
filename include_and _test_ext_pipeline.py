import spacy
from spacy.language import Language

custom_nlp = spacy.load("my_output/my_nlp", vocab=nlp.vocab)

# Usage as a decorator
@Language.factory(
   "custom_nlp",
   default_config={"some_setting": True},
)
def create_my_component(nlp, name, some_setting):
     return Custom_nlp(some_setting)

# Usage as function
Language.factory(
    "my_component",
    default_config={"some_setting": True},
    func=create_my_component
)



#import spacy

#nlp = spacy.load("en_core_web_lg")
#custom_nlp = spacy.load("my_output/my_nlp", vocab=nlp.vocab)

#nlp.add_pipe(custom_nlp.get_pipe("ner"), name="my_ner", before="ner")