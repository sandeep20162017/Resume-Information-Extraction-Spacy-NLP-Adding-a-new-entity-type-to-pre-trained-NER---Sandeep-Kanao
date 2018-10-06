## Python NLP - SpaCy - Adding a new entity type to an existing pre-trained NER - Sandeep Kanao

"""Example of training an additional entity type

This script shows how to add a new entity type to an existing pre-trained NER model.

The actual training is performed by looping over the examples, and calling `nlp.entity.update()`. The `update()` method steps through the words of the input.
At each word, it makes a prediction. It then consults the annotations provided on the GoldParse instance, to see whether it was right. If it was wrong, it adjusts its weights so that the correct action will score higher next time.

After training your model, you can save it to a directory. We recommend
wrapping models as Python packages, for ease of deployment.

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

"""
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy


# new entity label
LABEL_EDUCATION = 'EDUCATION'
LABEL_STUDIEDAT = 'STUDIEDAT'

# training data
# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting
TRAIN_DATA = [
    ("University", {
        'entities': [(0, 9, 'STUDIEDAT')]
    }),
    ("College", {
        'entities': [(0, 7, 'STUDIEDAT')]
    }),
    ("University Of Alberta", {
        'entities': [(0, 21, 'STUDIEDAT')]
    }),
    ("Ryerson University", {
        'entities': [(0, 18, 'STUDIEDAT')]
    }),
    ("Senaca College is based in Toronto", {
        'entities': [(0, 14, 'STUDIEDAT')]
    }),
    ("I Have Masters Degree in Computer Science", {
        'entities': [(7, 21, 'EDUCATION')]
    }),

    ("Do they bite?", {
        'entities': []
    }),

    ("Bachelor Of Computer Science", {
        'entities': [(0, 27, 'EDUCATION')]
    }),

    ("BSc", {
        'entities': [(0, 3, 'EDUCATION')]
    }),

    ("Bachelor", {
        'entities': [(0, 8, 'EDUCATION')]
    }),

    ("B.Sc.", {
        'entities': [(0, 5, 'EDUCATION')]
    })
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))

def main(model=None, new_model_name='resume', output_dir=None, n_iter=20):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe('ner')

    ner.add_label(LABEL_EDUCATION)   # add new entity label to entity recognizer
    ner.add_label(LABEL_STUDIEDAT)
    if model is None:
        optimizer = nlp.begin_training()
    else:
        # Note that 'begin_training' initializes the models, so it'll zero out
        # existing entity types.
        optimizer = nlp.entity.create_optimizer()

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update([text], [annotations], sgd=optimizer, drop=0.20,
                           losses=losses)
            print(losses)

    # test the trained model
    test_text = 'I studied Bachelor Of Computer Science at University Of Alberta'
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


#if __name__ == '__main__':
plac.call(main)