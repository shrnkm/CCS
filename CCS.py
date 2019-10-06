import fire
import json
import os
import nltk
import gensim
import numpy as np
import tensorflow as tf
from nltk import tokenize
from nltk.tokenize import RegexpTokenizer
from gpt2model import model, encoder, sample

# Mute tf WARNING messages
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load Google's pre-trained Word2Vec model.
Gmodel = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNewsModel/GoogleNews-vectors-negative300.bin', binary=True)



def interact_model(raw_text,
    model_name='774M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=40,
    top_p=1,
    models_dir='gpt2model',
):
    """
    Interactively run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)


        context_tokens = enc.encode(raw_text)
        generated = 0
        for _ in range(nsamples // batch_size):
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
            for i in range(batch_size):
                generated += 1
                text = enc.decode(out[i])
                # and here gpt2 returns the output
                print(text)

### Test the GPT2 Function
# print(interact_model('Cats are cute.'))



# ================================ Google W2V ==================================
# =================================Helper Functions=============================

# deal with stopwords and punctuation marks
stopwords_list = nltk.corpus.stopwords.words('english')
tokenizer = RegexpTokenizer(r'\w+')


# function to convert a sentence to a list of words
def sent_to_list(sent):
    '''
    gets sentence returns a list of words
    '''
    # tokenize the input sentence to a list
    sent_list = tokenizer.tokenize(sent)
    # and remove the stopwords
    sent_list = [word for word in sent_list if (not(word in stopwords_list) and word in Gmodel.vocab)]
    return sent_list


# function to convert a paragraph to a list of sentences
def text_to_sents(text):
    '''
    gets text returns a list of sentences
    '''
    # tokenize the gpt2 output
    sentences = tokenize.sent_tokenize(text.lower()) # to sentences
    return sentences



def get_candidates(text, goal_word):
    '''
    this function gets a paragraph and compare each of its sentences to a goal
    word and returns a dictionary of sentences and their similarity scores
    '''
    # if input is a string, split it into sentences
    if isinstance(text, str):
        text = text_to_sents(text)
    candidates = {}
    # similarity between current sentence and goal_word (must be updated)
    current_sim = 0.0
    for sent in text:
        sim = Gmodel.n_similarity(sent_to_list(sent), [goal_word])
        print(sim)
        if sim > current_sim:
            candidates[sent] = sim
            current_sim = sim

    return candidates


print(interact_model('icture yourself driving down a city street. You go around a curve, and suddenly see something in the middle of the road ahead. What should you do? Of course, the answer depends on what that  something  is. A torn paper bag, a lost shoe, or a tumbleweed? You can drive right over it without a second thought, but you ll definitely swerve around a pile of broken glass. You ll probably stop for a dog standing in the road but move straight into a flock of pigeons, knowing that the birds will fly out of the way. You might plough right through a pile of snow, but veer around a carefully constructed snowman. In short, you ll quickly determine the actions that best fit the situation – what humans call having  common sense. Human drivers aren t the only ones who need common sense; its lack in artificial intelligence (AI) systems will likely be the major obstacle to the wide deployment of fully autonomous cars. Even the best of today s self-driving cars are challenged by the object-in-the-road problem. Perceiving  obstacles  that no human would ever stop for, these vehicles are liable to slam on the brakes unexpectedly, catching other motorists off-guard. Rear-ending by human drivers is the most common accident involving self-driving cars. The challenges for autonomous vehicles probably won t be solved by giving cars more training data or explicit rules for what to do in unusual situations. To be trustworthy, these cars need common sense: broad knowledge about the world and an ability to adapt that knowledge in novel circumstances. While today s AI systems have made impressive strides in domains ranging from image recognition to language processing, their lack of a robust foundation of common sense makes them susceptible to unpredictable and unhumanlike errors. Common sense is multifaceted, but one essential aspect is the mostly tacit  core knowledge  that humans share – knowledge we are born with or learn by living in the world. That includes vast knowledge about the properties of objects, animals, other people and society in general, and the ability to flexibly apply this knowledge in new situations. You can predict, for example, that while a pile of glass on the road won t fly away as you approach, a flock of birds likely will. If you see a ball bounce in front of your car, for example, you know that it might be followed by a child or a dog running to retrieve it. From this perspective, the term  common sense  seems to capture exactly what current AI cannot do: use general knowledge about the world to act outside prior training or pre-programmed rules. Today s most successful AI systems use deep neural networks. These are algorithms trained to spot patterns, based on statistics gleaned from extensive collections of human-labelled examples. This process is very different from how humans learn. We seem to come into the world equipped with innate knowledge of certain basic concepts that help to bootstrap our way to understanding –  including the notions of discrete objects and events, the three-dimensional nature of space, and the very idea of causality itself. Humans also seem to be born with nascent concepts of sociality: babies can recognise simple facial expressions, they have inklings about language and its role in communication, and rudimentary strategies to entice adults into communication. Such knowledge is so elemental and immediate that we aren t even conscious we have it, or that it forms the basis for all future learning. A big lesson from decades of AI research is how hard it is to teach such concepts to machines.'))
