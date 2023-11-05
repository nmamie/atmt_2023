import argparse
import collections
import logging
import os
import sys
import re
import pickle

# establish link to seq2seq dir
# scripts_dir = os.path.dirname(os.path.abspath(__file__))
# base_dir = os.path.join(scripts_dir, "..")
# sys.path.append(base_dir)

from seq2seq import utils
from seq2seq.data.dictionary import Dictionary
from bpe.learn_bpe import *
from bpe.apply_bpe import *

SPACE_NORMALIZER = re.compile("\s+")


def word_tokenize(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def get_args():
    parser = argparse.ArgumentParser('Data pre-processing)')
    parser.add_argument('--source-lang', default=None, metavar='SRC', help='source language')
    parser.add_argument('--target-lang', default=None, metavar='TGT', help='target language')

    parser.add_argument('--train-prefix', default=None, metavar='FP', help='train file prefix')
    parser.add_argument('--tiny-train-prefix', default=None, metavar='FP', help='tiny train file prefix')
    parser.add_argument('--valid-prefix', default=None, metavar='FP', help='valid file prefix')
    parser.add_argument('--test-prefix', default=None, metavar='FP', help='test file prefix')
    parser.add_argument('--dest-dir', default='data-bin', metavar='DIR', help='destination dir')

    parser.add_argument('--threshold-src', default=2, type=int,
                        help='map words appearing less than threshold times to unknown')
    parser.add_argument('--bpe-train', action='store_true', help='train BPE model')
    parser.add_argument('--bpe-size', default=10000, type=int, help='number of BPE merge operations')
    parser.add_argument('--bpe-dropout', default=0.1, type=float, help='BPE dropout rate')
    parser.add_argument('--num-words-src', default=-1, type=int, help='number of source words to retain')
    parser.add_argument('--threshold-tgt', default=2, type=int,
                        help='map words appearing less than threshold times to unknown')
    parser.add_argument('--num-words-tgt', default=-1, type=int, help='number of target words to retain')
    parser.add_argument('--vocab-src', default=None, type=str, help='path to dictionary')
    parser.add_argument('--vocab-trg', default=None, type=str, help='path to dictionary')
    parser.add_argument('--quiet', action='store_true', help='no logging')

    return parser.parse_args()


def main(args):
    os.makedirs(args.dest_dir, exist_ok=True)
    
    if args.bpe_train:
        print(args.train_prefix + '.' + args.source_lang)
        train_bpe_model(args.train_prefix + '.' + args.source_lang, args.bpe_size, args.bpe_dropout)
    
    if not args.vocab_src:
        src_dict = build_dictionary([args.train_prefix + '.' + args.source_lang])

        src_dict.finalize(threshold=args.threshold_src, num_words=args.num_words_src)
        src_dict.save(os.path.join(args.dest_dir, 'dict.' + args.source_lang))
        if not args.quiet:
            logging.info('Built a source dictionary ({}) with {} words'.format(args.source_lang, len(src_dict)))

    else:
        src_dict = Dictionary.load(args.vocab_src)
        if not args.quiet:
            logging.info('Loaded a source dictionary ({}) with {} words'.format(args.target_lang, len(src_dict)))

    if not args.vocab_trg:
        tgt_dict = build_dictionary([args.train_prefix + '.' + args.target_lang])

        tgt_dict.finalize(threshold=args.threshold_tgt, num_words=args.num_words_tgt)
        tgt_dict.save(os.path.join(args.dest_dir, 'dict.' + args.target_lang))
        if not args.quiet:
            logging.info('Built a target dictionary ({}) with {} words'.format(args.target_lang, len(tgt_dict)))

    else:
        tgt_dict = Dictionary.load(args.vocab_trg)
        if not args.quiet:
            logging.info('Loaded a target dictionary ({}) with {} words'.format(args.target_lang, len(tgt_dict)))

    def make_split_datasets(lang, dictionary):
        if args.train_prefix is not None:
            make_binary_dataset(args.train_prefix + '.' + lang, os.path.join(args.dest_dir, 'train.' + lang),
                                dictionary)
        if args.tiny_train_prefix is not None:
            make_binary_dataset(args.tiny_train_prefix + '.' + lang, os.path.join(args.dest_dir, 'tiny_train.' + lang),
                                dictionary)
        if args.valid_prefix is not None:
            make_binary_dataset(args.valid_prefix + '.' + lang, os.path.join(args.dest_dir, 'valid.' + lang),
                                dictionary)
        if args.test_prefix is not None:
            make_binary_dataset(args.test_prefix + '.' + lang, os.path.join(args.dest_dir, 'test.' + lang), dictionary)

    make_split_datasets(args.source_lang, src_dict)
    make_split_datasets(args.target_lang, tgt_dict)


def train_bpe_model(train_file, bpe_size, bpe_dropout, num_symbols=10000, min_frequency=2, verbose=False, is_dict=False, total_symbols=False, num_workers=1):
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    newdir = os.path.join(currentdir, 'subword_nmt')

    if num_workers <= 0:
        num_workers = cpu_count()

    # read/write files as UTF-8
    if args.input.name != '<stdin>':
        args.input = codecs.open(args.input.name, encoding='utf-8')
    if args.output.name != '<stdout>':
        args.output = codecs.open(args.output.name, 'w', encoding='utf-8')

    learn_bpe(args.input, args.output, args.symbols, args.min_frequency, args.verbose, is_dict=args.dict_input, total_symbols=args.total_symbols, num_workers=args.num_workers)

    # close files
    if args.input.name != '<stdin>':
        args.input.close()
    if args.output.name != '<stdout>':
        args.output.close()
    


def bpe_tokenize(line):
    global bpe
       


def build_dictionary(filenames, tokenize=bpe_tokenize):
    dictionary = Dictionary()
    for filename in filenames:
        with open(filename, 'r') as file:
            for line in file:
                for symbol in tokenize(line.strip()):
                    dictionary.add_word(symbol)
                dictionary.add_word(dictionary.eos_word)
    return dictionary


def make_binary_dataset(input_file, output_file, dictionary, tokenize=bpe_tokenize, append_eos=True):
    nsent, ntok = 0, 0
    unk_counter = collections.Counter()

    def unk_consumer(word, idx):
        if idx == dictionary.unk_idx and word != dictionary.unk_word:
            unk_counter.update([word])

    tokens_list = []
    with open(input_file, 'r') as inf:
        for line in inf:
            tokens = dictionary.binarize(line.strip(), tokenize, append_eos, consumer=unk_consumer)
            nsent, ntok = nsent + 1, ntok + len(tokens)
            tokens_list.append(tokens.numpy())

    with open(output_file, 'wb') as outf:
        pickle.dump(tokens_list, outf, protocol=pickle.DEFAULT_PROTOCOL)
        if not args.quiet:
            logging.info('Built a binary dataset for {}: {} sentences, {} tokens, {:.3f}% replaced by unknown token'.format(
            input_file, nsent, ntok, 100.0 * sum(unk_counter.values()) / ntok, dictionary.unk_word))


if __name__ == '__main__':
    args = get_args()
    if not args.quiet:
        utils.init_logging(args)
        logging.info('COMMAND: %s' % ' '.join(sys.argv))
        logging.info('Arguments: {}'.format(vars(args)))
    main(args)
    
    
    
# python preprocess_bpe.py --source-lang fr \
# --target-lang en \
# --train-prefix data/en-fr/raw/train \
# --tiny-train-prefix data/en-fr/raw/tiny_train \
# --valid-prefix data/en-fr/raw/valid \
# --test-prefix data/en-fr/raw/test \
# --dest-dir data/en-fr/preprocessed \
# --bpe-train