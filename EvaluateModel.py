import spacy
from spacy.scorer import Scorer
from spacy.tokens import Doc
import random
from spacy.training.example import Example
from spacy.util import minibatch

test_data = [
("Better results are obtained when Neptune's eccentricity is excited to e<SUB>N</SUB> ≃ 0.1 and subsequently damped by dynamical friction.", {"entities": [(33, 40, 'UAT')]}),
('Integral field spectroscopy can map astronomical objects spatially and spectroscopically.', {"entities": [(15, 27, 'UAT')]}),
('The simulation also produces the observed postfall equinox cooling followed by rapid warming in the upper stratosphere.', {"entities": [(106, 118, 'UAT')]}),
]
nlp = spacy.load("my_output/my_nlp")#('en_core_web_sm') # for spaCy's pretrained use 'en_core_web_sm'

examples = []
for text, annots in test_data:
    examples.append(Example.from_dict(nlp.make_doc(text), annots))

for i in range(20):
    random.shuffle(examples)
    for batch in minibatch(examples, size=8):
        nlp.update(batch)

eval_result = nlp.evaluate(examples)

print(eval_result)