import numpy as np
from numpy import ndarray
from typing import Tuple, List, Iterable, Dict, Any
import re
import nltk
import tqdm
from . import utils as U
import logging
import codecs
from sklearn.metrics import cohen_kappa_score as QWK

logger = logging.getLogger(__name__)


def normalize_scores(scores: ndarray, score_range: Tuple) -> ndarray:
    '''
    Convert scores to boundary of [0, 1].
    arg scores_array: ndarray, scores to convert.
    return: ndarray, converted score array
    '''
    scores_array = scores
    if type(scores_array) == list:
        scores_array = np.array(scores_array)
    low, high = score_range
    scores_array = (scores_array - low) / (high - low)
    assert np.all(scores_array >= 0), f'{scores_array.min()} < 0'
    assert np.all(scores_array <= 1), f'{scores_array.max()} > 1'
    return scores_array


def recover_scores(scores: ndarray, score_range: Tuple) -> ndarray:
    '''
    Convert scores of essays to origin range.
    arg scores_array: ndarray, scores to convert.
    return: ndarray, converted score array
    '''
    scores_array = scores
    if type(scores_array) == list:
        scores_array = np.array(scores_array)
    assert np.all(scores_array >= 0), f'{scores_array.min()} < 0'
    assert np.all(scores_array <= 1), f'{scores_array.max()} > 1'
    low, high = score_range
    scores_array = scores_array * (high - low) + low
    assert np.all(scores_array >= low), f'{scores_array.min()} < {low}'
    assert np.all(scores_array <= high), f'{scores_array.max()} > {high}'
    return scores_array


def tokenize(string: str) -> List:
    # add space between special characters and word/numbers
    string = re.sub(r'([\W\d]+)', r' \1 ', string)
    tokens = nltk.word_tokenize(string)
    # for index, token in enumerate(tokens):
    #     # seems here recoganize some specific token in form of @abc
    #     # instead of @ and abc, but remove subsequence starting with
    #     # digitals from [abc]
    #     # 190401: find twitter ids?
    #     if token == '@' and (index+1) < len(tokens):
    #         tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
    #         tokens.pop(index)
    return tokens


def numerical(tokens: Iterable, vocab: Dict[str, int]) -> List[int]:
    return [vocab[t] for t in tokens]


def sent_to_sequence(
        sent: List[str], vocab: Dict[str, int], hits: Dict[str, int]
        ) -> ndarray:
    indices = list()
    for word in sent:
        if U.is_number(word):
            indices.append(vocab['<num>'])
            hits['num'] += 1
        elif word == '<pad>':
            indices.append(vocab['<pad>'])
            hits['pad'] += 1
        elif word in vocab:
            indices.append(vocab[word])
        else:
            indices.append(vocab['<unk>'])
            hits['unk'] += 1
    return np.array(indices)


def sents_to_sequences(
        sents: List[Any], vocab: Dict[str, int]
        ) -> List[np.ndarray]:
    '''Convert sents/tokens into lists of integer numbers.
    :param tokens: List[str] or List[List[str]]
        A list of sentences or a list of tokens. If it's a list of
        sentences, each sentence will be tokenized first.
    :param vocab: Dict[str, int]
        A dict mapping tokens to indices.
    :return List[List[int]]:
    '''
    data = []
    total = 0
    hits = {'num': 0, 'unk': 0, 'pad': 0}

    for i, sent in tqdm.tqdm(list(enumerate(sents))):
        if type(sents[0]) == str:
            sent = tokenize(sent)
        indices = sent_to_sequence(sent, vocab, hits)
        # for word in sent:
            # if U.is_number(word):
            #     indices.append(vocab['<num>'])
            #     hits['num'] += 1
            # elif word == '<pad>':
            #     indices.append(vocab['<pad>'])
            #     hits['pad'] += 1
            # elif word in vocab:
            #     indices.append(vocab[word])
            # else:
            #     indices.append(vocab['<unk>'])
            #     hits['unk'] += 1
        total += len(sent)
        data.append(np.array(indices))

    if total == 0:
        total = 1
    logger = logging.getLogger()
    for h in hits:
        logger.info(f'{h} hit rate: {100*hits[h]/total:.2f}%')
    return data


class EmbReader:
    def __init__(self, emb_path, emb_dim=None):
        logger.info('Loading embeddings from: ' + emb_path)
        has_header = False
        with codecs.open(emb_path, 'r', encoding='utf8') as emb_file:
            tokens = emb_file.readline().split()
            if len(tokens) == 2:
                try:
                    int(tokens[0])
                    int(tokens[1])
                    has_header = True
                except ValueError:
                    pass
        if has_header:
            with codecs.open(emb_path, 'r', encoding='utf8') as emb_file:
                tokens = emb_file.readline().split()
                assert len(tokens) == 2, \
                    'The first line in W2V embeddings must ' \
                    'be the pair (vocab_size, emb_dim)'
                self.vocab_size = int(tokens[0])
                self.emb_dim = int(tokens[1])
                assert self.emb_dim == emb_dim,\
                    f'The embeddings dimension {self.emb_dim} does not ' \
                    f'match with the requested dimension({emb_dim})'
                self.embeddings = {}
                counter = 0
                for line in emb_file:
                    line = line.rstrip()
                    tokens = line.split(' ')
                    assert len(tokens) == self.emb_dim + 1, \
                        f'#dimensions ({len(tokens)-1}) does not match to '\
                        f'the header info ({self.emb_dim})\n'\
                        f'next line: {emb_file.readline()}'
                    word = tokens[0]
                    vec = tokens[1:]
                    self.embeddings[word] = vec
                    counter += 1
                assert counter == self.vocab_size, \
                    'Vocab size does not match the header info'
        else:
            with codecs.open(emb_path, 'r', encoding='utf8') as emb_file:
                self.vocab_size = 0
                self.emb_dim = -1
                self.embeddings = {}
                for line in emb_file:
                    tokens = line.split()
                    if self.emb_dim == -1:
                        self.emb_dim = len(tokens) - 1
                        assert self.emb_dim == emb_dim, 'The embeddings dimension \
                            does not match with the requested dimension'
                    else:
                        assert len(tokens) == self.emb_dim + 1, 'The number of dimensions \
                            does not match the header info'
                    word = tokens[0]
                    vec = tokens[1:]
                    self.embeddings[word] = vec
                    self.vocab_size += 1

        logger.info(f'  #vec: {self.vocab_size}, #dim: {self.emb_dim}')

    def get_emb_given_word(self, word):
        try:
            return self.embeddings[word]
        except KeyError:
            return None

    def get_emb_matrix_given_vocab(self, vocab):
        counter = 0.
        emb_matrix = np.zeros((len(vocab), self.emb_dim))
        for word, index in vocab.items():
            try:
                emb_matrix[index] = self.embeddings[word]
                counter += 1
            except KeyError:
                pass
        rate = 100*counter/len(vocab)
        logger.info(f'{counter}/{len(vocab)} word \
            vectors initialized (hit rate: {rate:.2})')
        return emb_matrix

    def get_emb_dim(self):
        return self.emb_dim


def qwk(pred, y):
    return QWK(
        y.astype(int), np.rint(pred).astype(int), labels=None, weights='quadratic'
    )