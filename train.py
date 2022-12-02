import os
import json
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from collections import OrderedDict
from model import Generator, Discriminator
from preprocess import CustomDataset
from multiprocessing import cpu_count
import numpy as np
from rollout import Rollout
from scheduler import CosineWithWarmRestarts
from torch.autograd import Variable
from data import generate_data

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (step-x0))))
    elif anneal_function == 'linear':
        return min(1, step / x0)

def kl_loss_function(mean, logv, anneal_function, step, k, x0):
    KLLoss = (-0.5) * torch.sum(1 + logv - mean.pow(2) - logv.exp())
    KLWeight = kl_anneal_function(anneal_function, step, k, x0)

    return KLLoss, KLWeight

def nll_loss_function(NLL, logits, target):
    # Flatten 'target'
    target = target.contiguous().view(-1)

    # print(target.shape)

    # Resize prediction from 3D to 2D
    logits = logits.view(-1, logits.size(-1))
    # print(logits.shape)

    # Negative Log-likelihood Loss
    NLLLoss = NLL(logits, target.long())

    return NLLLoss

def nll_policy_loss(logits, target, rewards):
    """
    logits: (N, C), torch Variable
    target: (N, ), torch Variable
    rewards: (N, ), torch Variable 
    """
    # target: [128, 20]
    # logits: [128. 20, 30000]
    target = target.contiguous().view(-1) # [2560]
    logits = logits.view(-1, logits.size(-1)) # [2560, 30000]

    N = target.size(0)
    C = logits.size(1)
    one_hot = torch.zeros((N, C))
    if logits.is_cuda:
        one_hot = one_hot.cuda()
    one_hot.scatter_(1, target.data.view((-1,1)), 1)
    one_hot = one_hot.type(torch.BoolTensor)
    one_hot = Variable(one_hot)
    if logits.is_cuda:
        one_hot = one_hot.cuda()
    loss = torch.masked_select(logits, one_hot)
    loss = loss * rewards
    loss = -torch.sum(loss) / N
    return loss

def get_length(train):
    for i, _ in enumerate(train):
        pass
    return i

def main(args):
    device = args.device
        
    types = ['train', 'valid']

    datasets = OrderedDict()

    for typee in types:
        datasets[typee] = CustomDataset(
            filename=f"{typee}.txt",
            data_dir=args.data_dir,
            data_file=f"{typee}.json",
            file_type=typee,
            max_sequence_length=args.max_sequence_length,
            new_data=False,
        )

    params_generator = dict(
        vocab_size=datasets['train'].vocab_size,
        sos_idx=datasets['train'].sos_idx,
        eos_idx=datasets['train'].eos_idx,
        pad_idx=datasets['train'].pad_idx,
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        hidden_size=args.generator_hidden_size,
        latent_size=args.generator_latent_size,
        num_layers=args.generator_num_layers,
        bidirectional=args.bidirectional,
        device=device
    )

    params_discriminator = dict(
        vocab_size=datasets['train'].vocab_size,
        sos_idx=datasets['train'].sos_idx,
        eos_idx=datasets['train'].eos_idx,
        pad_idx=datasets['train'].pad_idx,
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        hidden_size=args.discriminator_hidden_size,
        num_layers=args.discriminator_num_layers,
        bidirectional=args.bidirectional,
        device=device
    )

    generator = Generator(**params_generator)
    discriminator = Discriminator(**params_discriminator)
    adversarial_loss = torch.nn.BCELoss()
    
    if args.pretrained_model:
        print("Loading model from: ", os.path.join(args.save_model_path, args.pretrained_path))
        generator.load_state_dict(torch.load(os.path.join(args.save_model_path, args.pretrained_path)))
        print("Finished.")

    if device in ["cuda", "cuda:1"]:
        generator = generator.to(device)
        discriminator = discriminator.to(device)
        adversarial_loss = adversarial_loss.to(device)

    save_model_path = args.save_model_path
    if os.path.exists(save_model_path) is False:
        os.makedirs(save_model_path)

    with open(os.path.join(save_model_path, 'model_params.json'), 'w') as f:
        json.dump(params_generator, f, indent=4)
        json.dump(params_discriminator, f, indent=4)

    log_likelihood_loss = torch.nn.NLLLoss(ignore_index=datasets['train'].pad_idx)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.learning_rate)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate)

    step = 0
    w2_origin = args.w2
    w3_origin = args.w3
    t_wp = args.t_wp

    train_iter = DataLoader(
                dataset=datasets['train'],
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )
    train_len = get_length(train_iter)

    if args.scheduler == True:
        sched_g = CosineWithWarmRestarts(optimizer_g, T_max=train_len)
        sched_d = CosineWithWarmRestarts(optimizer_d, T_max=train_len)

    min_current_loss = float("inf")

    if os.path.exists(os.path.join(args.logdir, "log.txt")):
        os.remove(os.path.join(args.logdir, "log.txt"))

    f_log = open(os.path.join(args.logdir, "log.txt"), "w")
    
    num_early_stopping = 0

    Tensor = torch.cuda.FloatTensor if device in ["cuda", "cuda:1"] else torch.FloatTensor
    w2 = w2_origin
    w3 = w3_origin

    print('Pretraining Generator with MLE...')
    for epoch in range(args.pre_num_epoch_gen):
        for typee in types:
            print(f"\n{typee.upper()}: Epoch {epoch+1}/{args.pre_num_epoch_gen}:")
            
            # Generating new data every epoch
            if typee == "train":
                generate_data(test_split=args.test_split, mode="train")
            
            datasets[typee] = CustomDataset(
                filename=f"{typee}.txt",
                data_dir=args.data_dir,
                data_file=f"{typee}.json",
                file_type=typee,
                max_sequence_length=args.max_sequence_length,
                new_data=(typee=="train"),
            )
        
            data_loader = DataLoader(
                dataset=datasets[typee],
                batch_size=args.batch_size,
                shuffle=(typee=='train'),
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )

            list_of_loss = []

            if typee == 'train':
                generator.train()
            else:
                generator.eval()

            for iteration, batch in enumerate(data_loader):
                batch_size = batch['origin'].size(0)

                for k, v in batch.items():
                    batch[k] = v.to(device)

                _, logits_1, logits_2, _, _ = generator(batch['origin'], batch['paraphrase'], batch['input_paraphrase'])

                nll_loss_1 = nll_loss_function(log_likelihood_loss, logits_1, batch['target'])
                nll_loss_2 = nll_loss_function(log_likelihood_loss, logits_2, batch['target'])

                loss = nll_loss_1 + nll_loss_2
                list_of_loss.append(loss.item())
                
                if typee == "train":
                    optimizer_g.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), args.g_norm)
                    optimizer_g.step()

                if typee == 'train' and (iteration + 1) == len(data_loader):
                    print("TRAIN: Epoch %.0f Summary: Loss: %.5f" % (epoch+1, np.mean(list_of_loss)))

        print("VALID: Epoch %.0f Summary: Loss: %.5f" % (epoch+1, np.mean(list_of_loss)))

        if typee == 'valid':
            mean_loss = np.mean(list_of_loss)
            print("Mean Generator Loss: %.5f" % mean_loss)
            
            if mean_loss < min_current_loss:
                min_current_loss = mean_loss
                num_early_stopping = 0
                checkpoint_path = os.path.join(save_model_path, "generator")

                torch.save(generator.state_dict(), checkpoint_path)
                print("Best model saved at: %s" % checkpoint_path)
                # f_log.write("Best model saved at: %s\n" % checkpoint_path)
    
    print('\nPretraining Discriminator...')
    min_current_loss = float("inf")

    for epoch in range(args.pre_num_epoch_dis):
        for typee in types:
            print(f"\n{typee.upper()}: Epoch {epoch+1}/{args.pre_num_epoch_gen}:")
        
            # Generating new data every epoch
            if typee == "train":
                generate_data(test_split=args.test_split, mode="train")

            datasets[typee] = CustomDataset(
                filename=f"{typee}.txt",
                data_dir=args.data_dir,
                data_file=f"{typee}.json",
                file_type=typee,
                max_sequence_length=args.max_sequence_length,
                new_data=(typee=="train"),
            )
            
            data_loader = DataLoader(
                dataset=datasets[typee],
                batch_size=args.batch_size,
                shuffle=(typee=='train'),
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )
            
            list_of_loss = []
                
            for iteration, batch in enumerate(data_loader):
                batch_size = batch['origin'].size(0)
                real = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False).to(device)
                fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False).to(device)

                for k, v in batch.items():
                    batch[k] = v.to(device)
                # batch = {k: v.to(device) for k, v in batch.items()}

                prediction = generator.inference(batch["origin"], device=device)
                real_loss = adversarial_loss(discriminator(batch['target']), real)
                fake_loss = adversarial_loss(discriminator(prediction), fake)

                loss = (real_loss + fake_loss) / 2
                list_of_loss.append(loss.item())

                if typee == "train":
                    optimizer_d.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.d_norm)
                    optimizer_d.step()

                if typee == "train" and (iteration + 1) == len(data_loader):
                    print("TRAIN: Epoch %.0f Summary: Loss: %.5f" % (epoch+1, np.mean(list_of_loss)))

        print("VALID: Epoch %.0f Summary: Loss: %.5f" % (epoch+1, np.mean(list_of_loss)))
        
        if typee == 'valid':
            mean_loss = np.mean(list_of_loss)
            print("Mean Discriminator Loss: %.5f" % mean_loss)
            
            if mean_loss < min_current_loss:
                min_current_loss = mean_loss
                num_early_stopping = 0
                checkpoint_path = os.path.join(save_model_path, "discriminator")

                torch.save(discriminator.state_dict(), checkpoint_path)
                print("Best model saved at: %s" % checkpoint_path)
                # f_log.write("Best model saved at: %s\n" % checkpoint_path)
    
    print("Adversarial Training with Policy Gradient...")
    generate_data(test_split=args.test_split, mode="train")
    rollout = Rollout(generator, args.update_rate, device)
    min_current_loss = float("inf")

    print("Loading Generator from: ", os.path.join(save_model_path, "generator"))
    generator.load_state_dict(torch.load(os.path.join(save_model_path, "generator")))
    print("Finished.")

    print("Loading Discriminator from: ", os.path.join(save_model_path, "discriminator"))
    discriminator.load_state_dict(torch.load(os.path.join(save_model_path, "discriminator")))
    print("Finished.")
    
    for epoch in range(args.epochs):
        for typee in types:
            list_of_e1_loss = []
            list_of_e2_loss = []
            list_of_g_loss = []
            list_of_d_loss = []

            print(f"\n{typee.upper()}: Epoch {epoch+1}/{args.epochs}:")
            f_log.write(f"\n{typee.upper()}: Epoch {epoch+1}/{args.epochs}:\n")

            data_loader = DataLoader(
                dataset=datasets[typee],
                batch_size=args.batch_size,
                shuffle=(typee=='train'),
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )

            if typee == 'train':
                generator.train()
                discriminator.train()
            else:
                generator.eval()
                discriminator.eval()

            if typee == 'train':
                print("Generator...")
                f_log.write("Generator...\n")
                for iteration, batch in enumerate(data_loader):
                    batch_size = batch['origin'].size(0)
                    real = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False).to(device)

                    for k, v in batch.items():
                        batch[k] = v.to(device)

                    if step < t_wp:
                        w2 = (step / t_wp) * w2
                        w3 = (step / t_wp) * w3
                    else:
                        w2 = w2_origin
                        w3 = w3_origin
                    
                    w1 = 1 - w2

                    pred_2, logits_1, logits_2, mean, logv = generator(batch['origin'], batch['paraphrase'], batch['input_paraphrase'])
                    
                    rewards = rollout.get_reward(pred_2, args.roll_out_loop, discriminator)
                    rewards = Variable(torch.Tensor(rewards))
                    rewards = torch.exp(rewards).contiguous().view((-1,))
                    if device in ["cuda", "cuda:1"]:
                        rewards = rewards.cuda()
                    
                    KLLoss, KLWeight = kl_loss_function(mean, logv, 
                                                    args.anneal_function, step, 
                                                    args.k, args.x0)

                    kl_loss = (KLWeight * KLLoss) / batch_size

                    nll_loss_1 = nll_policy_loss(logits_1, batch['target'], rewards)
                    nll_loss_2 = nll_policy_loss(logits_2, batch['target'], rewards)
                    # print(pred_2.device, real.device)
                    dg_loss = adversarial_loss(discriminator(pred_2), real)

                    e1_loss = w1 * kl_loss + w1 * nll_loss_1 + w2 * nll_loss_2 + w3 * dg_loss
                    e2_loss = kl_loss + nll_loss_1
                    g_loss = w1 * nll_loss_1 + w2 * nll_loss_2 + w3 * dg_loss

                    list_of_e1_loss.append(e1_loss.item())
                    list_of_e2_loss.append(e2_loss.item())
                    list_of_g_loss.append(g_loss.item())

                    optimizer_g.zero_grad()
                    e1_loss.backward(retain_graph=True)
                    e2_loss.backward(retain_graph=True)
                    g_loss.backward()

                    torch.nn.utils.clip_grad_norm_(generator.parameters(), args.g_norm)
                    optimizer_g.step()

                    if args.scheduler == True: 
                        sched_g.step()

                    step += 1

                    if iteration % args.print_every == 0 or (iteration + 1) == len(data_loader):
                        print("Batch: %i/%i, E1_Loss: %.5f, E2_Loss: %.5f, G_Loss: %.5f" % 
                            (iteration, len(data_loader)-1, e1_loss.item(), e2_loss.item(), g_loss.item()))

                        f_log.write("Batch: %i/%i, E1_Loss: %.5f, E2_Loss: %.5f, G_Loss: %.5f\n" % 
                            (iteration, len(data_loader)-1, e1_loss.item(), e2_loss.item(), g_loss.item()))

                        if (iteration + 1) == len(data_loader):
                            print("Summary: E1_Loss: %.5f, E2_Loss: %.5f, G_Loss: %.5f" % 
                                (np.mean(list_of_e1_loss), np.mean(list_of_e2_loss), np.mean(list_of_g_loss)))

                            f_log.write("Summary: E1_Loss: %.5f, E2_Loss: %.5f, G_Loss: %.5f\n" % 
                                (np.mean(list_of_e1_loss), np.mean(list_of_e2_loss), np.mean(list_of_g_loss)))
                
                rollout.update_params()

                # Discriminator
                print("Discriminator...")
                f_log.write("Discriminator...\n")
                for iteration, batch in enumerate(data_loader):
                    batch_size = batch['origin'].size(0)
                    real = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False).to(device)
                    fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False).to(device)

                    for k, v in batch.items():
                        batch[k] = v.to(device)

                    prediction = generator.inference(batch["origin"], device=device)

                    real_loss = adversarial_loss(discriminator(batch['target']), real)
                    fake_loss = adversarial_loss(discriminator(prediction), fake)
                    d_loss = (real_loss + fake_loss) / 2

                    list_of_d_loss.append(d_loss.item())

                    optimizer_d.zero_grad()
                    d_loss.backward()

                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.d_norm)
                    optimizer_d.step()

                    if args.scheduler == True: 
                        sched_d.step()

                    if iteration % args.print_every == 0 or (iteration + 1) == len(data_loader):
                        print("Batch: %i/%i, D_Loss: %.5f" % (iteration, len(data_loader)-1, d_loss.item()))
                        f_log.write("Batch: %i/%i, D_Loss: %.5f\n" % (iteration, len(data_loader)-1, d_loss.item()))

                        if (iteration + 1) == len(data_loader):
                            print("Summary: D_Loss: %.5f" % (np.mean(list_of_d_loss)))
                            f_log.write("Summary: D_Loss: %.5f\n" % (np.mean(list_of_d_loss)))
            else:
                for iteration, batch in enumerate(data_loader):
                    batch_size = batch['origin'].size(0)
                    real = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False).to(device)
                    fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False).to(device)

                    for k, v in batch.items():
                        batch[k] = v.to(device)
                    
                    pred_2, logits_1, logits_2, mean, logv = generator(batch['origin'], batch['paraphrase'], batch['input_paraphrase'])
                    
                    KLLoss, KLWeight = kl_loss_function(mean, logv, 
                                                    args.anneal_function, step, 
                                                    args.k, args.x0)

                    kl_loss = (KLWeight * KLLoss) / batch_size

                    nll_loss_1 = nll_loss_function(log_likelihood_loss, logits_1, batch['target'])
                    nll_loss_2 = nll_loss_function(log_likelihood_loss, logits_2, batch['target'])
                    dg_loss = adversarial_loss(discriminator(pred_2), real)

                    e1_loss = w1 * kl_loss + w1 * nll_loss_1 + w2 * nll_loss_2 + w3 * dg_loss
                    e2_loss = kl_loss + nll_loss_1
                    g_loss = w1 * nll_loss_1 + w2 * nll_loss_2 + w3 * dg_loss

                    list_of_e1_loss.append(e1_loss.item())
                    list_of_e2_loss.append(e2_loss.item())
                    list_of_g_loss.append(g_loss.item())

                    real_loss = adversarial_loss(discriminator(batch['target']), real)
                    fake_loss = adversarial_loss(discriminator(pred_2), fake)
                    d_loss = (real_loss + fake_loss) / 2

                    list_of_d_loss.append(d_loss.item())

                    if (iteration + 1) == len(data_loader):
                        print("Summary: E1_Loss: %.5f, E2_Loss: %.5f, G_Loss: %.5f, D_Loss: %.5f" % 
                            (np.mean(list_of_e1_loss), np.mean(list_of_e2_loss), np.mean(list_of_g_loss), np.mean(list_of_d_loss)))

                        f_log.write("Summary: E1_Loss: %.5f, E2_Loss: %.5f, G_Loss: %.5f, D_Loss: %.5f\n" % 
                            (np.mean(list_of_e1_loss), np.mean(list_of_e2_loss), np.mean(list_of_g_loss), np.mean(list_of_d_loss)))

            # Save checkpoint
            if typee == 'valid':
                mean_loss = np.mean(list_of_e1_loss)
                print("Mean Generator Loss: %.5f" % mean_loss)
                
                if mean_loss < min_current_loss:
                    min_current_loss = mean_loss
                    num_early_stopping = 0
                    checkpoint_path = os.path.join(save_model_path, "model")

                    torch.save(generator.state_dict(), checkpoint_path)
                    print("Best model saved at: %s" % checkpoint_path)
                    f_log.write("Best model saved at: %s\n" % checkpoint_path)

                    prediction = generator.inference(batch["origin"], device=device)
                    print("Prediction ", prediction)
                    print("Target ", batch["target"])
                elif mean_loss > min_current_loss and args.early_stopping == True:
                    num_early_stopping += 1
                    if num_early_stopping == args.num_early_stopping:
                        print(f"Early Stopping after {args.num_early_stopping} epochs!")
                        f_log.write("Early Stopping after %s epochs!" % args.num_early_stopping)
                        raise Exception

            f_log.write("\n")

    f_log.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--max_sequence_length', type=int, default=20)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=bool, default=True)
    parser.add_argument('--device', type=str, default="cuda")

    parser.add_argument('--embedding_size', type=int, default=300)

    parser.add_argument('--generator_hidden_size', type=int, default=512)
    parser.add_argument('--generator_latent_size', type=int, default=300)

    parser.add_argument('--discriminator_hidden_size', type=int, default=512)

    parser.add_argument('--g_norm', type=int, default=10)
    parser.add_argument('--d_norm', type=int, default=5)

    parser.add_argument('--generator_num_layers', type=int, default=1)
    parser.add_argument('--discriminator_num_layers', type=int, default=1)
    parser.add_argument('--bidirectional', type=bool, default=False)

    parser.add_argument('--anneal_function', type=str, default='logistic')
    parser.add_argument('--k', type=float, default=0.0032)
    parser.add_argument('--x0', type=int, default=3200)

    parser.add_argument('--w2', type=float, default=0.5)
    parser.add_argument('--w3', type=float, default=0.01)
    parser.add_argument('--t_wp', type=float, default=10000)

    parser.add_argument('--early_stopping', type=bool, default=True)
    parser.add_argument('--num_early_stopping', type=int, default=10)
    parser.add_argument('--print_every', type=int, default=100) # print every batch
    
    parser.add_argument('--logdir', type=str, default='logs/')
    parser.add_argument('--save_model_path', type=str, default='checkpoints/')

    parser.add_argument('--pretrained_path', type=str, default="old_model")
    parser.add_argument('--pretrained_model', type=bool, default=False)

    parser.add_argument('--test_split', type=float, default=0.2)
    parser.add_argument('--pre_num_epoch_gen', type=int, default=100)
    parser.add_argument('--pre_num_epoch_dis', type=int, default=5)

    parser.add_argument('--update_rate', type=int, default=0.8)
    parser.add_argument('--roll_out_loop', type=int, default=8)

    args = parser.parse_args()

    args.anneal_function = args.anneal_function.lower()

    assert args.anneal_function in ['logistic', 'linear']

    main(args)