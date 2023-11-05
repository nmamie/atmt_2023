import os
import argparse
import logging
import collections
import pickle
import numpy
import sentencepiece as spm

from seq2seq import utils
from seq2seq.data.dictionary import Dictionary

# Initialize SentencePiece processor
sp_processor = None

from bpe.lean_joint_bpe_and_vocab import *
from bpe.apply_bpe import *

def get_args():
    parser = argparse.ArgumentParser('Data pre-processing)')
    parser.add_argument('--source-lang', default=None, metavar='SRC', help='source language')
    parser.add_argument('--target-lang', default=None, metavar='TGT', help='target language')
    parser.add_argument('--train-prefix', default=None, metavar='FP', help='train file prefix')
    parser.add_argument('--tiny-train-prefix', default=None, metavar='FP', help='tiny train file prefix')
    parser.add_argument('--valid-prefix', default=None, metavar='FP', help='valid file prefix')
    parser.add_argument('--test-prefix', default=None, metavar='FP', help='test file prefix')
    parser.add_argument('--dest-dir', default='data-bin', metavar='DIR', help='destination dir')
    parser.add_argument('--threshold-src', default=2, type=int, help='map words appearing less than threshold times to unknown')
    parser.add_argument('--num-words-src', default=-1, type=int, help='number of source words to retain')
    parser.add_argument('--threshold-tgt', default=2, type=int, help='map words appearing less than threshold times to unknown')
    parser.add_argument('--num-words-tgt', default=-1, type=int, help='number of target words to retain')
    parser.add_argument('--vocab-src', default=None, type=str, help='path to dictionary')
    parser.add_argument('--vocab-trg', default=None, type=str, help='path to dictionary')
    parser.add_argument('--quiet', action='store_true', help='no logging')
    parser.add_argument('--bpe-model', default=None, type=str, help='Path to the trained SentencePiece BPE model.')
    parser.add_argument('--dropout-rate', default=0.1, type=float, help='Dropout rate for BPE dropout.')
    parser.add_argument('--train-bpe', action='store_true', help='Train a new BPE model')
    parser.add_argument('--vocab-size', default=1000 , help='Vocab dataset')
    parser.add_argument('--train-autoencoder', action='store_true', help='Use autoencoder to learn to copy monolingual data to source side')
    
    # bpe arguments
    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), required=True, nargs = '+',
        metavar='PATH',
        help="Input texts (multiple allowed).")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), required=True,
        metavar='PATH',
        help="Output file for BPE codes.")
    parser.add_argument(
        '--symbols', '-s', type=int, default=10000,
        help="Create this many new symbols (each representing a character n-gram) (default: %(default)s)")
    parser.add_argument(
        '--separator', type=str, default='@@', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s')")
    parser.add_argument(
        '--write-vocabulary', type=argparse.FileType('w'), required=True, nargs = '+', default=None,
        metavar='PATH', dest='vocab',
        help='Write to these vocabulary files after applying BPE. One per input text. Used for filtering in apply_bpe.py')
    parser.add_argument(
        '--min-frequency', type=int, default=2, metavar='FREQ',
        help='Stop if no symbol pair has frequency >= FREQ (default: %(default)s)')
    parser.add_argument(
        '--total-symbols', '-t', action="store_true",
        help="subtract number of characters from the symbols to be generated (so that '--symbols' becomes an estimate for the total number of symbols needed to encode text).")
    parser.add_argument(
        '--num-workers', type=int, default=16,
        help="Number of processors to process texts, only supported in Python3. If -1, set `multiprocessing.cpu_count()`. (default: %(default)s)")
    parser.add_argument(
        '--verbose', '-v', action="store_true",
        help="verbose mode.")

    return parser.parse_args()

def monolingual_corpus(src_file, tgt_file, output_file):
    """Create a mixed corpus based on 'Copied Monolingual Data Improves Low-Resource Neural Machine
    Translation (Currey et al., 2017)' that includes the original parallel training data as well as the 
    target data that is additionally copied to the source side."""
    
    # merge the source and target files
    with open(src_file, 'r') as src, open(tgt_file, 'r') as tgt, open(output_file, 'w') as out:
        for src_line, tgt_line in zip(src, tgt):
            out.write(src_line.strip() + '\n' + tgt_line.strip() + '\n')
        
    # close the files
    src.close()
    tgt.close()
    out.close()     

    
def bpe_tokenize(line, dropout_rate=0.1, source_lang=True):
    if source_lang:
        vocabulary = 'data/en-fr/preprocessed/vocab.fr'
    else:
        vocabulary = 'data/en-fr/preprocessed/vocab.en'
    bpe = BPE(codecs.open('data/en-fr/preprocessed/bpe_codes', encoding='utf-8'), separator='@@', vocab=vocabulary)
    return bpe.process_line(line, dropout_rate)    

def build_dictionary(filenames, source_lang=True, tokenize=bpe_tokenize):
    dictionary = Dictionary()
    for filename in filenames:
        with open(filename, 'r') as file:
            for line in file:
                # for symbol in tokenize(line.strip(), source_lang=source_lang):
                for symbol in line.strip().split(' '):
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

def main(args):
    os.makedirs(args.dest_dir, exist_ok=True)

    if args.train_bpe:
        # train_bpe_model(vocab_size=args.vocab_size)
        learn_joint_bpe_and_vocab(args)
        args.bpe_model = 'mymodel'
        
    if args.train_autoencoder:
        # train data
        monolingual_corpus(args.train_prefix + '.' + args.source_lang, args.train_prefix + '.' + args.target_lang, args.train_prefix + '.' + 'mono.' + args.source_lang)
        monolingual_corpus(args.train_prefix + '.' + args.target_lang, args.train_prefix + '.' + args.target_lang, args.train_prefix + '.' + 'mono.' + args.target_lang)
        args.train_prefix = args.train_prefix + '.' + 'mono'
        # tiny train data
        monolingual_corpus(args.tiny_train_prefix + '.' + args.source_lang, args.tiny_train_prefix + '.' + args.target_lang, args.tiny_train_prefix + '.' + 'mono.' + args.source_lang)
        monolingual_corpus(args.tiny_train_prefix + '.' + args.target_lang, args.tiny_train_prefix + '.' + args.target_lang, args.tiny_train_prefix + '.' + 'mono.' + args.target_lang)
        args.tiny_train_prefix = args.tiny_train_prefix + '.' + 'mono'

    if not args.vocab_src:
        src_dict = build_dictionary([args.train_prefix + '.bpe.' + args.source_lang], source_lang=True)
        src_dict.finalize(threshold=args.threshold_src, num_words=args.num_words_src)
        src_dict.save(os.path.join(args.dest_dir, 'dict.' + args.source_lang))
        if not args.quiet:
            logging.info('Built a source dictionary ({}) with {} words'.format(args.source_lang, len(src_dict)))
    else:
        src_dict = Dictionary.load(args.vocab_src)
        if not args.quiet:
            logging.info('Loaded a source dictionary ({}) with {} words'.format(args.source_lang, len(src_dict)))

    if not args.vocab_trg:
        trg_dict = build_dictionary([args.train_prefix + '.bpe.' + args.target_lang], source_lang=False)
        trg_dict.finalize(threshold=args.threshold_tgt, num_words=args.num_words_tgt)
        trg_dict.save(os.path.join(args.dest_dir, 'dict.' + args.target_lang))
        if not args.quiet:
            logging.info('Built a target dictionary ({}) with {} words'.format(args.target_lang, len(trg_dict)))
    else:
        trg_dict = Dictionary.load(args.vocab_trg)
        if not args.quiet:
            logging.info('Loaded a target dictionary ({}) with {} words'.format(args.target_lang, len(trg_dict)))

    make_binary_dataset(args.train_prefix + '.bpe.' + args.source_lang, os.path.join(args.dest_dir, 'train.' + args.source_lang), src_dict)
    make_binary_dataset(args.train_prefix + '.bpe.' + args.target_lang, os.path.join(args.dest_dir, 'train.' + args.target_lang), trg_dict)
    make_binary_dataset(args.valid_prefix + '.bpe.' + args.source_lang, os.path.join(args.dest_dir, 'valid.' + args.source_lang), src_dict)
    make_binary_dataset(args.valid_prefix + '.bpe.' + args.target_lang, os.path.join(args.dest_dir, 'valid.' + args.target_lang), trg_dict)
    make_binary_dataset(args.test_prefix + '.bpe.' + args.source_lang, os.path.join(args.dest_dir, 'test.' + args.source_lang), src_dict)
    make_binary_dataset(args.test_prefix + '.bpe.' + args.target_lang, os.path.join(args.dest_dir, 'test.' + args.target_lang), trg_dict)

if __name__ == '__main__':
    args = get_args()
    if not args.quiet:
        logging.basicConfig(format='%(message)s', level=logging.INFO)
    main(args)