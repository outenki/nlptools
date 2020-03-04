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
import operator

logger = logging.getLogger(__name__)


def normalize_scores(scores: ndarray, score_range: ndarray) -> ndarray:
    '''
    Convert scores to boundary of [0, 1].
    arg scores_array: ndarray, scores to convert.
    return: ndarray, converted score array
    '''
    scores_array = scores
    if type(scores_array) == list:
        scores_array = np.array(scores_array)
    if type(score_range) != ndarray:
        score_range = np.array(score_range)
    if len(score_range.shape) == 1:
        score_range = np.expand_dims(score_range, 0)
        score_range = score_range.repeat(len(scores), 0)
    low, high = score_range[:, 0], score_range[:, 1]
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
    if type(score_range) != ndarray:
        score_range = np.array(score_range)
    if len(score_range.shape) == 1:
        score_range = np.expand_dims(score_range, 0)
        score_range = score_range.repeat(len(scores), 0)
    assert np.all(scores_array >= 0), f'{scores_array.min()} < 0'
    assert np.all(scores_array <= 1), f'{scores_array.max()} > 1'
    # low, high = score_range
    low, high = score_range[:, 0], score_range[:, 1]
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


def tokens_to_indices(
        tokens: List[str], vocab: Dict[str, int], hits: Dict[str, int]
        ) -> ndarray:
    indices = list()
    if not hits:
        hits = {'num': 0, 'pad': 0, 'unk': 0}
    for word in tokens:
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
        indices = tokens_to_indices(sent, vocab, hits)
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
        y.astype(int), np.rint(pred).astype(int),
        labels=None, weights='quadratic'
    )


def create_vocab_from_tokens(
        tokens: List[str], vocab_size: int, to_lower: bool
        ):
    ''' Create vocabulary from tokens.
    Vocabulary is a dict mapping words to an integer (index).
    The words would be sorted by frequency.
    :param tokens: List[str], tokens.
    :param vocab_size: int, the max length of vocab, i.e. the number of keys
        of vocab dict
    :param to_lower: bool. All the tokens will be set to lower.
    :return: dict. A dict mapping tokens to integers. Sort tokens by frequency
        and take the indices as values of dict. There are special tokens of
        '<pad>', '<num>', '<unkown>' and '<dummy>', standing for 'paddings',
        'numbers', 'unkown words' and 'special functional token'.
    '''
    total_words, unique_words = 0, 0
    word_freqs = {}

    for word in tqdm.tqdm(tokens):
        try:
            word_freqs[word] += 1
        except KeyError:
            unique_words += 1
            word_freqs[word] = 1
        total_words += 1
    logger.info(f'{total_words} total words, {unique_words} unique words')
    print(f'{total_words} total words, {unique_words} unique words')

    # sort word_freqs by frequency
    sorted_word_freqs = sorted(
        word_freqs.items(), key=operator.itemgetter(1), reverse=True
    )
    if vocab_size <= 0:
        # Choose vocab size automatically by removing all singletons
        # Only the top-vocab_size words will be returned as vocab
        # If vocab_size was set as 0, then calculate it as the number of
        # words appeared more than once.
        vocab_size = len(sorted_word_freqs)
        # vocab_size = 0
        # for word, freq in sorted_word_freqs:
        #     if freq > 1:
        #         vocab_size += 1
    vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2, '<dummy>': 3}
    vcb_len = len(vocab)
    index = vcb_len
    for word, _ in sorted_word_freqs[:vocab_size - vcb_len]:
        vocab[word] = index
        index += 1
    return vocab
