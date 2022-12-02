import copy
import numpy as np
import torch

class Rollout(object):
    """Roll-out policy"""
    def __init__(self, model, update_rate, device):
        self.ori_model = model
        self.own_model = copy.deepcopy(model)
        self.update_rate = update_rate
        self.device = device
        self.Tensor = torch.cuda.FloatTensor if device in ["cuda", "cuda:1"] else torch.FloatTensor

    def get_reward(self, x, num, discriminator):
        """
        Args:
            x : (batch_size, seq_len)
            num : roll-out number
            discriminator : discrimanator model
        """
        rewards = []
        # batch_size = x.size(0)
        seq_len = x.size(1)
        for i in range(num):
            for l in range(1, seq_len):
                # data = x[:, 0:l]
                # samples = self.own_model.sample(batch_size, seq_len, data)
                samples = self.own_model.inference(x, device=self.device)
                pred = discriminator(samples)
                pred = pred.detach().cpu().numpy()
                if i == 0:
                    rewards.append(pred)
                else:
                    rewards[l-1] += pred

            # for the last token
            pred = discriminator(x)
            pred = pred.detach().cpu().numpy()
            if i == 0:
                rewards.append(pred)
            else:
                rewards[seq_len-1] += pred
        rewards = np.transpose(np.array(rewards)) / (1.0 * num)
        return rewards

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]