import os
import json
import torch
import argparse
from torch.utils.data import DataLoader
from collections import OrderedDict
from model import Generator
from preprocess import CustomDataset
from multiprocessing import cpu_count
from torchtext.data.metrics import bleu_score
# from nltk.translate.bleu_score import sentence_bleu

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def to_word(predictions, idx2word, word2idx):
    sent_str = [str()] * len(predictions)
    for i, sent in enumerate(predictions):
        for id in sent:
            if id == word2idx["<pad>"]:
                break
            sent_str[i] += idx2word[str(id.item())] + " "
    return sent_str

def main(args):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        
    typee = 'valid'

    datasets = OrderedDict()

    datasets[typee] = CustomDataset(
        filename=f"{typee}.txt",
        data_dir=args.data_dir,
        data_file=f"{typee}.json",
        file_type=typee,
        max_sequence_length=args.max_sequence_length,
        new_data=False,
    )
    
    with open(os.path.join(args.data_dir, 'vocab.json'), 'r') as file:
        vocab = json.load(file)
    file.close()

    word2idx, idx2word = vocab['word2idx'], vocab['idx2word']

    params = dict(
        vocab_size=len(word2idx),
        sos_idx=word2idx["<sos>"],
        eos_idx=word2idx["<eos>"],
        pad_idx=word2idx["<pad>"],
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        device=device
    )

    data_loader = DataLoader(
        dataset=datasets['valid'],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cpu_count(),
        pin_memory=torch.cuda.is_available()
    )

    if not os.path.exists(args.save_model_path):
        raise FileNotFoundError(args.save_model_path)
    
    model = Generator(**params)
    
    print("Loading model from: ", os.path.join(args.save_model_path))
    model.load_state_dict(torch.load(os.path.join(args.save_model_path)))
    print("Finished.")

    print(model)

    model = model.to(device)
    
    model.eval()      

    f = open("output.txt","w")
    f_1 = open("duplicate.txt", "w")

    bleu_pred = []
    bleu_targ = []

    for iteration, batch in enumerate(data_loader):

        for k, v in batch.items():
            batch[k] = v.to(device)

        prediction = model.inference(batch["origin"], device=device)
        list_of_inputs = to_word(batch["origin"], idx2word, word2idx)
        list_of_targets = to_word(batch["target"], idx2word, word2idx)
        list_of_predictions = to_word(prediction, idx2word, word2idx)

        for inp, pred, targ in zip(list_of_inputs, list_of_predictions, list_of_targets): #, list_of_targets):
            f.write(f"Inp: {inp}") # Input
            f.write("\n")
            f.write(f"Pre: {pred}") # Prediction
            f.write("\n")
            f.write(f"Tar: {targ}") # Target
            f.write("\n\n")

            if pred == targ:
                f_1.write(f"Inp: {inp}") # Input
                f_1.write("\n")
                f_1.write(f"Pre: {pred}") # Prediction
                f_1.write("\n")
                f_1.write(f"Tar: {targ}") # Target
                f_1.write("\n\n")
            
            bleu_pred.append(pred.split())
            bleu_targ.append([targ.split()])
            # print(sentence_bleu([targ.split()], pred.split()))
            # print(bleu_score([pred.split()], [[targ.split()]]))
            # print()

        # if iteration == 0:
        #     break

    bleu_s = bleu_score(bleu_pred, bleu_targ)
    print("BLEU Score: ", bleu_s)
    f.write(f"BLEU Score: {bleu_s}")

    f.close()
    f_1.close()
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_model_path', type=str, default='checkpoints/model')

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--max_sequence_length', type=int, default=20)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--bidirectional', type=bool, default=False)
    parser.add_argument('--latent_size', type=int, default=300)

    args = parser.parse_args()

    main(args)