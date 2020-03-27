import os
import unicodedata

from collections import OrderedDict

from transformers import BertTokenizer, WordpieceTokenizer
from transformers.tokenization_bert import load_vocab


class MecabBasicTokenizer(object):
    """Runs basic tokenization with MeCab morphological parser."""

    def __init__(self, do_lower_case=False, never_split=None,
                 mecab_dict_path=None, preserve_spaces=False):
        """Constructs a MecabBasicTokenizer.
        Args:
            **do_lower_case**: (`optional`) boolean (default True)
                Whether to lower case the input.
            **mecab_dict_path**: (`optional`) string
                Path to a directory of a MeCab dictionary.
            **preserve_spaces**: (`optional`) boolean (default True)
                Whether to preserve whitespaces in the output tokens.
        """
        if never_split is None:
            never_split = []

        self.do_lower_case = do_lower_case
        self.never_split = never_split

        import MeCab
        if mecab_dict_path is not None:
            self.mecab = MeCab.Tagger('-d {}'.format(mecab_dict_path))
        else:
            self.mecab = MeCab.Tagger()

        self.preserve_spaces = preserve_spaces

    def tokenize(self, text, never_split=None, with_info=False, **kwargs):
        """Tokenizes a piece of text."""
        never_split = self.never_split + \
            (never_split if never_split is not None else [])
        text = unicodedata.normalize('NFKC', text)

        tokens = []
        token_infos = []

        cursor = 0
        for line in self.mecab.parse(text).split('\n'):
            if line == 'EOS':
                if self.preserve_spaces and len(text[cursor:]) > 0:
                    tokens.append(text[cursor:])
                    token_infos.append(None)

                break

            token, token_info = line.split('\t')

            token_start = text.index(token, cursor)
            token_end = token_start + len(token)
            if self.preserve_spaces and cursor < token_start:
                tokens.append(text[cursor:token_start])
                token_infos.append(None)

            if self.do_lower_case and token not in never_split:
                token = token.lower()

            tokens.append(token)
            token_infos.append(token_info)

            cursor = token_end

        assert len(tokens) == len(token_infos)
        if with_info:
            return tokens, token_infos
        else:
            return tokens


class MecabBertTokenizer(BertTokenizer):
    """BERT tokenizer for Japanese text; MeCab tokenization + WordPiece"""

    def __init__(self,
                 vocab_file,
                 do_lower_case=False,
                 do_basic_tokenize=True,
                 do_wordpiece_tokenize=True,
                 mecab_dict_path=None,
                 unk_token='[UNK]',
                 sep_token='[SEP]',
                 pad_token='[PAD]',
                 cls_token='[CLS]',
                 mask_token='[MASK]',
                 **kwargs):
        """Constructs a MecabBertTokenizer.
        Args:
            **vocab_file**: Path to a one-wordpiece-per-line vocabulary file.
            **do_lower_case**: (`optional`) boolean (default True)
                Whether to lower case the input.
                Only has an effect when do_basic_tokenize=True.
            **do_basic_tokenize**: (`optional`) boolean (default True)
                Whether to do basic tokenization with MeCab before wordpiece.
            **mecab_dict_path**: (`optional`) string
                Path to a directory of a MeCab dictionary.
        """
        super(BertTokenizer, self).__init__(
            unk_token=unk_token, sep_token=sep_token, pad_token=pad_token,
            cls_token=cls_token, mask_token=mask_token, **kwargs)

        # take into account special tokens
        self.max_len_single_sentence = self.max_len - 2
        # take into account special tokens
        self.max_len_sentences_pair = self.max_len - 3

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'."
                .format(vocab_file))

        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        self.do_wordpiece_tokenize = do_wordpiece_tokenize

        if do_basic_tokenize:
            self.basic_tokenizer = MecabBasicTokenizer(
                do_lower_case=do_lower_case,
                mecab_dict_path=mecab_dict_path)

        if do_wordpiece_tokenize:
            self.wordpiece_tokenizer = WordpieceTokenizer(
                vocab=self.vocab,
                unk_token=self.unk_token)

    def _tokenize(self, text):
        if self.do_basic_tokenize:
            tokens = self.basic_tokenizer.tokenize(
                text, never_split=self.all_special_tokens
            )
        else:
            tokens = [text]

        if self.do_wordpiece_tokenize:
            split_tokens = [sub_token for token in tokens
                            for sub_token in
                            self.wordpiece_tokenizer.tokenize(token)]
        else:
            split_tokens = tokens

        return split_tokens
