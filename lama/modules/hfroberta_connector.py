# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from pytorch_transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM

import torch
import numpy as np
from lama.modules.base_connector import *

import torch.nn.functional as F

class HfRoberta(Base_Connector):

    def __init__(self, args):
        super().__init__()

        if args.hfroberta_model_dir is not None:
            # load bert model from file
            roberta_model_name = str(args.hfroberta_model_dir) + "/"
            dict_file = roberta_model_name
            print("loading huggingface RoBERTa model from {}".format(roberta_model_name))
        else:
            # load RoBERTa model from huggingface cache
            roberta_model_name = args.hfroberta_model_name
            dict_file = roberta_model_name

        # When using a cased model, make sure to pass do_lower_case=False directly to BaseTokenizer
        do_lower_case = False
        if 'uncased' in roberta_model_name:
            do_lower_case=True

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = RobertaTokenizer.from_pretrained(dict_file)

        # original vocab
        self.map_indices = None

        # GPT uses different way to represent BPE then BERT. Namely, the
        # final suffixes are indicated with </w> suffix, while pieces that must
        # be followed are written as is. In BERT the prefixes are written as is
        # while the parts that must follow (not be followed!) have '##' prefix.
        # There is no one-to-one coversion. But at least we may make pieces that
        # may form a full word look the same.
        # Note that we should be very careful now,
        # tokenizer.convert_tokens_to_ids won't work with our vocabulary.
        def convert_word(word):
            if word == ROBERTA_UNK:  # word == OPENAI_UNK:
                return word
            if word == '\n</w>':
                # Redefine symbol EOS to improve visualization.
                return ROBERTA_EOS  # OPENAI_EOS
            # return word[:-4] if word.endswith('</w>') else f'{word}##'
            return word[:-4] if word.endswith('</w>') else f'{word}'

        _, gpt_vocab = zip(*sorted(self.tokenizer.decoder.items()))
        self.vocab = [convert_word(word) for word in gpt_vocab]
        self._init_inverse_vocab()

        # Get UNK symbol as it's written in the origin RoBERTa vocab.
        unk_index = self.inverse_vocab[ROBERTA_UNK]  # OPENAI_UNK
        self.unk_symbol = self.tokenizer.decoder[unk_index]

        # Get MASK symbol as it's written in the origin RoBERTa vocab.
        mask_index = self.inverse_vocab[ROBERTA_MASK]
        self.mask_symbol = self.tokenizer.decoder[mask_index]

        # Load pre-trained model (weights)
        self.masked_roberta_model = RobertaForMaskedLM.from_pretrained(roberta_model_name)
        self.masked_roberta_model.eval()
        print(self.masked_roberta_model.config)

        # ... to get hidden states
        self.roberta_model = self.masked_roberta_model.roberta

        # Sanity check.
        #assert len(self.vocab) == self.masked_roberta_model.config.vocab_size
        #assert 0 == self.masked_roberta_model.config.n_special

        self.eos_id = self.inverse_vocab[ROBERTA_END_SENTENCE]  # OPENAI_EOS
        self.model_vocab = self.vocab

        self.pad_id = self.inverse_vocab[ROBERTA_PAD]
        self.unk_index = self.inverse_vocab[ROBERTA_UNK]
        self.mask_index = mask_index

    def __get_token_ids_from_tensor(self, indexed_string):
        token_ids = []
        if self.map_indices is not None:
            # map indices to subset of the vocabulary
            indexed_string = self.convert_ids(indexed_string)
            token_ids = np.asarray(indexed_string)
        else:
            token_ids = indexed_string
        return token_ids

    def _cuda(self):
        self.masked_roberta_model.cuda()

    def get_id(self, string):
        tokenized_text = self.tokenizer.tokenize(string)
        indexed_string = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        # indexed_string = self.convert_ids(indexed_string)
        return indexed_string

    def __get_input_tensors_batch(self, sentences_list):
        tokens_tensors_list = []
        segments_tensors_list = []
        masked_indices_list = []
        tokenized_text_list = []
        max_tokens = 0
        for sentences in sentences_list:
            tokens_tensor, segments_tensor, masked_indices, tokenized_text = self.__get_input_tensors(sentences)
            tokens_tensors_list.append(tokens_tensor)
            segments_tensors_list.append(segments_tensor)
            masked_indices_list.append(masked_indices)
            tokenized_text_list.append(tokenized_text)
            # assert(tokens_tensor.shape[1] == segments_tensor.shape[1])
            if (tokens_tensor.shape[1] > max_tokens):
                max_tokens = tokens_tensor.shape[1]
        # print("MAX_TOKENS: {}".format(max_tokens))
        # apply padding and concatenate tensors
        # use [PAD] for tokens and 0 for segments
        final_tokens_tensor = None
        final_segments_tensor = None
        final_attention_mask = None
        for tokens_tensor, segments_tensor in zip(tokens_tensors_list, segments_tensors_list):
            dim_tensor = tokens_tensor.shape[1]
            pad_lenght = max_tokens - dim_tensor
            attention_tensor = torch.full([1,dim_tensor], 1, dtype= torch.long)
            if pad_lenght>0:
                pad_1 = torch.full([1,pad_lenght], self.pad_id, dtype= torch.long)
                pad_2 = torch.full([1,pad_lenght], 0, dtype= torch.long)
                attention_pad = torch.full([1,pad_lenght], 0, dtype= torch.long)
                tokens_tensor = torch.cat((tokens_tensor,pad_1), dim=1)
                segments_tensor = torch.cat((segments_tensor,pad_2), dim=1)
                attention_tensor = torch.cat((attention_tensor,attention_pad), dim=1)
            if final_tokens_tensor is None:
                final_tokens_tensor = tokens_tensor
                final_segments_tensor = segments_tensor
                final_attention_mask = attention_tensor
            else:
                final_tokens_tensor = torch.cat((final_tokens_tensor,tokens_tensor), dim=0)
                final_segments_tensor = torch.cat((final_segments_tensor,segments_tensor), dim=0)
                final_attention_mask = torch.cat((final_attention_mask,attention_tensor), dim=0)
        # print(final_tokens_tensor)
        # print(final_segments_tensor)
        # print(final_attention_mask)
        # print(final_tokens_tensor.shape)
        # print(final_segments_tensor.shape)
        # print(final_attention_mask.shape)
        return final_tokens_tensor, final_segments_tensor, final_attention_mask, masked_indices_list, tokenized_text_list

    def __get_input_tensors(self, sentences):
        tokenized_text = []
        masked_indices = []
        segment_indices = []
        for sentence_idx, sentence in enumerate(sentences):
            if sentence_idx > 0:
                tokenized_text.append(ROBERTA_END_SENTENCE) # OPENAI_EOS)
            for chunk_idx, chunk in enumerate(sentence.split('[MASK]')):
                if chunk_idx > 0:
                    masked_indices.append(len(tokenized_text))
                    segment_indices.append(sentence_idx)
                    tokenized_text.append(self.mask_symbol)
                chunk = chunk.strip()
                if chunk:
                    tokenized_sentence = self.tokenizer.tokenize(chunk)
                    segment_id = np.full(len(tokenized_sentence),
                                         sentence_idx,
                                         dtype=int).tolist()

                    tokenized_text.extend(tokenized_sentence)
                    segment_indices.extend(segment_id)

        # add [CLS] token at the beginning
        tokenized_text.insert(0,ROBERTA_START_SENTENCE)
        segment_indices.insert(0,0)

        # look for masked indices
        masked_indices = []
        for i in range(len(tokenized_text)):
            token = tokenized_text[i]
            if token == ROBERTA_MASK:  # MASK
                masked_indices.append(i)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segment_indices])

        return tokens_tensor, segments_tensors, masked_indices, tokenized_text


    def get_batch_generation(self, sentences_list, logger=None, try_cuda=True):
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokens_tensor, segments_tensor, attention_mask_tensor, masked_indices_list, tokenized_text_list = self.__get_input_tensors_batch(sentences_list)

        if logger is not None:
            logger.debug("\n{}\n".format(tokenized_text_list))

        with torch.no_grad():
            logits = self.masked_roberta_model(
                input_ids=tokens_tensor.to(self._model_device),
                token_type_ids=segments_tensor.to(self._model_device),
                attention_mask=attention_mask_tensor.to(self._model_device),
            )
            if isinstance(logits, tuple):  # ケースによって、tupleだったり、そうでなかったり..
                logits = logits[0]

            log_probs = F.log_softmax(logits, dim=-1).cpu()

        token_ids_list = []
        for indexed_string in tokens_tensor.numpy():
            token_ids_list.append(self.__get_token_ids_from_tensor(indexed_string))

        return log_probs, token_ids_list, masked_indices_list

    def get_contextual_embeddings(self, sentences_list, try_cuda=True):

        # assume in input 1 or 2 sentences - in general, it considers only the first 2 sentences
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokens_tensor, segments_tensor, attention_mask_tensor, masked_indices_list, tokenized_text_list = self.__get_input_tensors_batch(sentences_list)

        with torch.no_grad():
            all_encoder_layers, _ = self.roberta_model(
                tokens_tensor.to(self._model_device),
                segments_tensor.to(self._model_device))

        all_encoder_layers = [layer.cpu() for layer in all_encoder_layers]

        sentence_lengths = [len(x) for x in tokenized_text_list]

        # all_encoder_layers: a list of the full sequences of encoded-hidden-states at the end
        # of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
        # encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size]
        return all_encoder_layers, sentence_lengths, tokenized_text_list