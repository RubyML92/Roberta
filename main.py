import torch
import argparse
import random
import numpy as np
from tokenization import Tokenizer
from roberta import ModelConfig, Roberta

def main():
    parser = argparse.ArgumentParser()
    # Other parameters
    parser.add_argument("--load_model", default=None, type=str, help="The pre-trained model to load")
    parser.add_argument("--save_model", default='BaseSave', type=str, help="The name that the models save as")
    parser.add_argument("--config", default='roberta_config.json', help="The config of roberta model")
    parser.add_argument("--seed", default=123, type=int, help="random seeed for initialization")
    parser.add_argument("--gpu_id", default=1, type=int, help="id of gpu")
    parser.add_argument("--vocab_file", default=None, type=str, help="Vocab txt for vocabulary")


    args = parser.parse_args()

    if torch.cuda.is_available():
        print("cuda {} is available".format(args.gpu_id))
        device = torch.device("cuda", args.gpu_id) #
        n_gpu = 1
    else:
        device = None
        print("cuda is unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = Tokenizer(args.vocab_file)

    config = ModelConfig.from_json_file(args.config)
    model = Roberta(config, tokenizer.vocab, device)

    load_model_file = args.load_model+".bin" if args.load_model else None
    if load_model_file and os.path.exists(load_model_file):
        model_dic = torch.load(load_model_file)
        model.load_state_dict(model_dic, strict=True)
        print("successfully load pre-trained model ...")
    elif config.method in ['Roberta']:
        model_dic = torch.load('data/roberta-base-pytorch_model.bin')
        model.load_state_dict(model_dic, strict=False)
        print("successfully load Roberta model ...")
    else:
        print("successfully initialize model ...")

    if args.gpu_id >= 0:   ###
        model.to(device)

    sent = ['They spent a great deal of money.', 'Is it the Great Wall in China?']
    sq_lens, sq = [], []
    for s in sent:
        tokens = tokenizer.tokenize(s)
        sq_lens.append(min(20, len(tokens)))         ### use len in bi-gru
        sq.append(tokenizer.convert_tokens_to_ids(tokens)[:20])
    # print('sq_lens: ', sq_lens)
    mcl = np.min([max(sq_lens)+2, 20])
    # print('mcl: ', mcl)
    sequence, sequence_position = [], []
    for s in sq:
        sequence += [[config.bos_token_id] + s + [config.eos_token_id]]
        sequence_position += [list(range(len(s)+2))]
    sequence = truncated_sequence(sequence, mcl)
    sequence_position = truncated_sequence(sequence_position, mcl)
    # print('sequence: ', sequence)
    sequence = torch.tensor(sequence, dtype=torch.long)
    # print('sequence shape: ', sequence.shape)
    sequence_position = torch.tensor(sequence_position, dtype=torch.long)
    # print('sequence_position shape: ', sequence_position.shape)

    batch = sequence, sequence_position
    if args.gpu_id >= 0: ready_batch = tuple(t.to(device) for t in batch)
    # exit()
    outputs = model(ready_batch)
    # print('outputs shape: ', outputs.shape)
    # print('outputs: ', outputs)

def truncated_sequence(sequence, mcl, fill=0):
    for c_idx, c in enumerate(sequence):
        if len(c) > mcl:
            sequence[c_idx] = c[:mcl]
        elif len(c) < mcl:
            sequence[c_idx] += [fill] * (mcl - len(c))
    return sequence

if __name__ == '__main__':
    main()
