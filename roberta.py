import torch
import torch.nn as nn

from RoBertaModel import *
import math
import numpy as np
import json
import zipfile
import os
import copy


class RobertaModel(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.device = device
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.pooler = RobertaPooler(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        output_hidden_states=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify input_ids")

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=self.device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        # print('embedding_output shape: ', embedding_output.shape) ## (batch, s_len, hidden_size)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states
        )

        sequence_output = encoder_outputs[-1]
        # print('sequence_output shape: ', sequence_output.shape) ## (batch, s_len, hidden_size)

        pooled_output = self.pooler(sequence_output)
        # print('pooled_output shape: ', pooled_output.shape) ## (batch, hidden_size)
        # exit()
        return pooled_output

class ModelConfig(object):
    """Configuration class to store the configuration of a 'Model'
    """
    def __init__(self,
                vocab_size_or_config_json_file,
                hidden_size = 200,
                dropout_prob = 0.1,
                initializer_range= 0.02):

        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r") as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.dropout_prob = dropout_prob
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """COnstruct a 'Config' from a Python dictionary of parameters."""
        config = ModelConfig(vocab_size_or_config_json_file = -1)
        for key, value in json_object.items():
            config.__dict__[key]=value
        return config
    @classmethod
    def from_json_file(cls, json_file):
        """Construct a 'Config' from a json file of parameters"""
        with open(json_file, 'r') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary"""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string"""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class RobertaPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, vocab, *input, **kwargs):
        super(RobertaPreTrainedModel, self).__init__()
        if not isinstance(config, ModelConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config
        self.vocab = vocab

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, (nn.Embedding)) and self.config.hidden_size==300 and module.weight.data.size(0) > 1000:
            if os.path.exists(self.config.Word2vec_path):
                embedding = np.load(self.config.Word2vec_path)
                module.weight.data = torch.tensor(embedding, dtype=torch.float)
                print('pretrained GloVe embeddings')
            else:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                gloves = zipfile.ZipFile(self.config.glove_file)
                seen = 0

                for glove in gloves.infolist():
                    with gloves.open(glove) as f:
                        for line in f:
                            if line != "":
                                splitline = line.split()
                                word = splitline[0].decode('utf-8')
                                embedding = splitline[1:]

                            if word in self.vocab and len(embedding) == 300:
                                temp = np.array([float(val) for val in embedding])
                                module.weight.data[self.vocab[word], :] = torch.tensor(temp, dtype=torch.float)
                                seen += 1

                print('pretrianed vocab %s among %s' %(seen, len(self.vocab)))
                np.save(self.config.Word2vec_path, module.weight.data.numpy())
        # elif isinstance(module, BertLayerNorm):
        #     module.bias.data.normal_(mean=0.0, std=self.config.initializer_range)
        #     module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class Roberta(RobertaPreTrainedModel):
    def __init__(self, config, vocab, device):
        super(Roberta, self).__init__(config, vocab, device)
        self.device = device
        self.config = config
        self.rbtmodel = RobertaModel(config, device=device)
        self.apply(self.init_weights)

    def forward(self, batch):
        sequence, sequence_position = batch  # (batch, sl)
        outputs = self.rbtmodel(input_ids=sequence,
                            position_ids=sequence_position,
                            output_hidden_states=self.config.output_hidden_states)
        return outputs
