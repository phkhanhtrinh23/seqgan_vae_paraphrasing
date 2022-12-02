import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, latent_size, max_sequence_length,
                sos_idx, eos_idx, pad_idx, device, num_layers=1, bidirectional=True):
        super().__init__()

        if torch.cuda.is_available():
            self.tensor = torch.cuda.FloatTensor
        else:
            self.tensor = torch.Tensor

        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.max_sequence_length = max_sequence_length
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.device = device

        self.embedding_o = nn.Embedding(
                            num_embeddings = self.vocab_size,
                            embedding_dim = self.embedding_size
                            )
        
        self.embedding_p = nn.Embedding(
                            num_embeddings = self.vocab_size,
                            embedding_dim = self.embedding_size
                            )

        self.encoder_1 = nn.LSTM(
                        input_size = self.embedding_size,
                        hidden_size = self.hidden_size,
                        num_layers = self.num_layers,
                        bidirectional = self.bidirectional,
                        batch_first = True
                        )

        self.encoder_2 = nn.LSTM(
                        input_size = self.embedding_size,
                        hidden_size = self.hidden_size,
                        num_layers = self.num_layers,
                        bidirectional = self.bidirectional,
                        batch_first = True
                        )
        
        self.generator_1 = nn.LSTM(
                        input_size = self.embedding_size,
                        hidden_size = self.hidden_size,
                        num_layers = self.num_layers,
                        bidirectional = self.bidirectional,
                        batch_first = True
                        )

        self.generator_2 = nn.LSTM(
                        input_size = self.embedding_size,
                        hidden_size = self.hidden_size,
                        num_layers = self.num_layers,
                        bidirectional = self.bidirectional,
                        batch_first = True
                        )
        
        if self.bidirectional:
            self.hidden_dim = 2 * self.num_layers
        else:
            self.hidden_dim = self.num_layers
        
        self.hidden_to_mean = nn.Linear(self.hidden_size * self.hidden_dim, self.latent_size)
        self.hidden_to_log_variance = nn.Linear(self.hidden_size * self.hidden_dim, self.latent_size)
        self.latent_to_hidden = nn.Linear(self.latent_size, self.hidden_size * self.hidden_dim)

        self.origin_to_hidden = nn.Linear(self.embedding_size, self.hidden_size * self.hidden_dim)
        self.squeeze_origin = nn.Linear(self.max_sequence_length, 1)

        if self.bidirectional:
            self.output_to_vocab = nn.Linear(self.hidden_size * 2, self.vocab_size)
        else:
            self.output_to_vocab = nn.Linear(self.hidden_size, self.vocab_size)
    
    def forward(self, origin_sequence, paraphrase_sequence, input_paraphrase):
        # ===== Encoder =====
        batch_size = origin_sequence.size(0)
        origin_embedding = self.embedding_o(origin_sequence) 
        paraphrase_embedding = self.embedding_p(paraphrase_sequence)
        input_paraphrase_embedding = self.embedding_p(input_paraphrase)

        _, (hidden_state_e1, cell_state_e1) = self.encoder_1(origin_embedding)
        _, (hidden_state_e, _) = self.encoder_2(paraphrase_embedding, (hidden_state_e1, cell_state_e1))

        # ===== z2 from E1 =====
        hidden_state_e1 = hidden_state_e1.view(batch_size, self.hidden_size * self.hidden_dim) 
        
        mean = self.hidden_to_mean(hidden_state_e1)
        log_variance = self.hidden_to_log_variance(hidden_state_e1)
        std = torch.exp(0.5 * log_variance)

        z2 = mean + std * torch.randn([batch_size, self.latent_size]).to(self.device)

        # ===== z1 from E =====
        hidden_state_e = hidden_state_e.view(batch_size, self.hidden_size * self.hidden_dim) 
        
        mean = self.hidden_to_mean(hidden_state_e)
        log_variance = self.hidden_to_log_variance(hidden_state_e)
        std = torch.exp(0.5 * log_variance)

        z1 = mean + std * torch.randn([batch_size, self.latent_size]).to(self.device)

        # ===== Conditional input =====
        # origin_hidden = self.origin_to_hidden(origin_embedding)
        # origin_hidden = origin_hidden.view(batch_size, self.hidden_size * self.hidden_dim, self.max_sequence_length)
        # origin_hidden = self.squeeze_origin(origin_hidden)
        # origin_hidden = origin_hidden.view(batch_size, self.hidden_size * self.hidden_dim)

        # ===== Generator =====
        _, (hidden_state_g1, cell_state_g1) = self.generator_1(origin_embedding)
        
        # ===== z1 =====
        hidden_state_z1 = self.latent_to_hidden(z1)
        hidden_state_z1 = hidden_state_z1.view(self.hidden_dim, batch_size, self.hidden_size)

        hidden_state_z1_g1 = hidden_state_z1 + hidden_state_g1
        cell_state_z1_g1 = hidden_state_z1 + cell_state_g1

        output_z1_g1, (_, _) = self.generator_2(input_paraphrase_embedding, (hidden_state_z1_g1, cell_state_z1_g1))
        output_z1_g1 = self.output_to_vocab(output_z1_g1)
        logits_1 = nn.functional.log_softmax(output_z1_g1, dim=-1)

        # ===== z2 =====
        hidden_state_z2 = self.latent_to_hidden(z2)
        hidden_state_z2 = hidden_state_z2.view(self.hidden_dim, batch_size, self.hidden_size)

        hidden_state_z2_g1 = hidden_state_z2 + hidden_state_g1
        cell_state_z2_g1 = hidden_state_z2 + cell_state_g1

        output_z2_g1, (_, _) = self.generator_2(input_paraphrase_embedding, (hidden_state_z2_g1, cell_state_z2_g1))
        output_z2_g1 = self.output_to_vocab(output_z2_g1)
        logits_2 = nn.functional.log_softmax(output_z2_g1, dim=-1)

        prediction_2 = self.predict(origin_sequence, z2, hidden_state_z2_g1, cell_state_z2_g1, self.device)
        prediction_2 = prediction_2.to(self.device)

        # prediction_2, logits_2 = self.predict(paraphrase_sequence, z2, hidden_state_z2_g1, cell_state_z2_g1, self.device)

        # _, logits_1 = self.inference(
        #                 batch_size=batch_size,
        #                 hidden_state_g1=hidden_state_g1,
        #                 cell_state_g1=cell_state_g1,
        #                 z=z_1,
        #                 device=self.device,
        #                 )

        # prediction_2, logits_2 = self.inference(
        #                 batch_size=batch_size,
        #                 hidden_state_g1=hidden_state_g1,
        #                 cell_state_g1=cell_state_g1,
        #                 z=z_2,
        #                 device=self.device,
        #                 )

        return prediction_2, logits_1, logits_2, mean, log_variance
    
    def _save_prediction(self, prediction, indices, sequence_update, t):
        temp = prediction[sequence_update]
        # update token at position t
        temp[:,t] = indices.data
        prediction[sequence_update] = temp
        return prediction

    # def _save_prediction_logit(self, logits, logit, sequence_update, t):
    #     logits[sequence_update, t, :] = logit
    #     return logits
    
    def predict(self, origin_sequence, z, hidden_state_z_g, cell_state_z_g, device="cpu"):
        # ===== Encoder =====
        batch_size = origin_sequence.size(0)

        # list of indices in a batch_size of sentences => choose which sentence to updates at timestep t
        sequence_idx = torch.arange(0, batch_size).long().to(device)
        sequence_update = torch.arange(0, batch_size).long().to(device)
        sequence_mask = torch.ones(batch_size).bool().to(device)
        running_sequence = torch.arange(0, batch_size).long().to(device)

        prediction = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()
        # logits = self.tensor(batch_size, self.max_sequence_length, self.vocab_size).fill_(self.pad_idx).to(device)

        hidden_state = hidden_state_z_g
        cell_state = cell_state_z_g
        # z = z.unsqueeze(1)

        t = 0

        while t < self.max_sequence_length and len(running_sequence) > 0:
            if t == 0:
                decoder_input_sequence = self.tensor(batch_size).fill_(self.sos_idx).long().to(device)
            """
            Ex:
            >>> batch_size = 10
            >>> self.sos_idx = 2
            >>> decoder_input_sequence = self.tensor(batch_size).fill_(self.sos_idx).long()
            >>> decoder_input_sequence
            tensor([2,2,2,..,2])
            >>> decoder_input_sequence.shape
            torch.Size([10])
            """

            decoder_input_sequence = decoder_input_sequence.unsqueeze(1)

            """
            Ex:
            >>> decoder_input_sequence = decoder_input_sequence.unsqueeze(1)
            >>> decoder_input_sequence
            tensor([[2],
                    [2],
                    [2],
                    [2],
                    [2],
                    [2],
                    [2],
                    [2],
                    [2],
                    [2]])
            >>> decoder_input_sequence.shape
            torch.Size([10, 1])
            """
            decoder_input_embedding = self.embedding_p(decoder_input_sequence)
            # decoder_input_embedding = decoder_input_embedding + z
            
            # inference
            output, (hidden_state, cell_state) = self.generator_2(decoder_input_embedding, (hidden_state, cell_state))
            output = self.output_to_vocab(output)
            # logit = torch.log_softmax(output, dim=-1)
            # logit = logit.squeeze(1)

            # return list of indices with shape = (batch_size, 1) with each row is top k index
            _, indices = torch.topk(output, k=1, dim=-1)
            indices = indices.reshape(-1)
            
            prediction = self._save_prediction(prediction, indices, sequence_update, t)
            # logits = self._save_prediction_logit(logits, logit, sequence_update, t)

            # masked_select: pick True index
            sequence_mask[sequence_update] = (indices != self.eos_idx)
            sequence_update = sequence_idx.masked_select(sequence_mask)

            running_mask = (indices != self.eos_idx).data
            running_sequence = torch.masked_select(running_sequence, running_mask)

            if len(running_sequence) > 0:
                decoder_input_sequence = indices[running_sequence]
                z = z[running_sequence]
                hidden_state = hidden_state[:, running_sequence]
                cell_state = cell_state[:, running_sequence]

                running_sequence = torch.arange(0, len(running_sequence)).long().to(device)

            t += 1
        
        return prediction#, logits
    
    def inference(self, origin_sequence, device="cuda"):
        # ===== Encoder =====
        batch_size = origin_sequence.size(0)
        origin_embedding = self.embedding_o(origin_sequence)

        # origin_hidden = self.origin_to_hidden(origin_embedding)
        # origin_hidden = origin_hidden.view(batch_size, self.hidden_size * self.hidden_dim, self.max_sequence_length)
        # origin_hidden = self.squeeze_origin(origin_hidden)
        # origin_hidden = origin_hidden.view(batch_size, self.hidden_size * self.hidden_dim)

        _, (hidden_state_e1, _) = self.encoder_1(origin_embedding)

        hidden_state_e1 = hidden_state_e1.view(batch_size, self.hidden_size * self.hidden_dim) 
        
        mean = self.hidden_to_mean(hidden_state_e1)
        log_variance = self.hidden_to_log_variance(hidden_state_e1)
        std = torch.exp(0.5 * log_variance)

        z2 = mean + std * torch.randn([batch_size, self.latent_size]).to(device)

        hidden_state_z2 = self.latent_to_hidden(z2)
        hidden_state_z2 = hidden_state_z2.view(self.hidden_dim, batch_size, self.hidden_size)

        _, (hidden_state_g1, cell_state_g1) = self.generator_1(origin_embedding)

        hidden_state_z2_g1 = hidden_state_z2 + hidden_state_g1
        cell_state_z2_g1 = hidden_state_z2 + cell_state_g1

        # list of indices in a batch_size of sentences => choose which sentence to updates at timestep t
        sequence_idx = torch.arange(0, batch_size).long().to(device)
        sequence_update = torch.arange(0, batch_size).long().to(device)
        sequence_mask = torch.ones(batch_size).bool().to(device)
        running_sequence = torch.arange(0, batch_size).long().to(device)

        prediction = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        hidden_state = hidden_state_z2_g1
        cell_state = cell_state_z2_g1
        # z2 = z2.unsqueeze(1)

        t = 0

        while t < self.max_sequence_length and len(running_sequence) > 0:
            if t == 0:
                decoder_input_sequence = self.tensor(batch_size).fill_(self.sos_idx).long().to(device)
            """
            Ex:
            >>> batch_size = 10
            >>> self.sos_idx = 2
            >>> decoder_input_sequence = self.tensor(batch_size).fill_(self.sos_idx).long()
            >>> decoder_input_sequence
            tensor([2,2,2,..,2])
            >>> decoder_input_sequence.shape
            torch.Size([10])
            """

            decoder_input_sequence = decoder_input_sequence.unsqueeze(1)

            """
            Ex:
            >>> decoder_input_sequence = decoder_input_sequence.unsqueeze(1)
            >>> decoder_input_sequence
            tensor([[2],
                    [2],
                    [2],
                    [2],
                    [2],
                    [2],
                    [2],
                    [2],
                    [2],
                    [2]])
            >>> decoder_input_sequence.shape
            torch.Size([10, 1])
            """
            decoder_input_embedding = self.embedding_p(decoder_input_sequence)
            # decoder_input_embedding = decoder_input_embedding + z2
            
            # inference
            output, (hidden_state, cell_state) = self.generator_2(decoder_input_embedding, (hidden_state, cell_state))
            output = self.output_to_vocab(output)

            # return list of indices with shape = (batch_size, 1) with each row is top k index
            _, indices = torch.topk(output, k=1, dim=-1)
            indices = indices.reshape(-1)
            
            prediction = self._save_prediction(prediction, indices, sequence_update, t)

            # masked_select: pick True index
            sequence_mask[sequence_update] = (indices != self.eos_idx)
            sequence_update = sequence_idx.masked_select(sequence_mask)

            running_mask = (indices != self.eos_idx).data
            running_sequence = torch.masked_select(running_sequence, running_mask)

            if len(running_sequence) > 0:
                decoder_input_sequence = indices[running_sequence]
                z2 = z2[running_sequence]
                hidden_state = hidden_state[:, running_sequence]
                cell_state = cell_state[:, running_sequence]

                running_sequence = torch.arange(0, len(running_sequence)).long().to(device)

            t += 1
        
        return prediction

class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, max_sequence_length,
                sos_idx, eos_idx, pad_idx, device, num_layers=1, bidirectional=True):
        super().__init__()

        if torch.cuda.is_available():
            self.tensor = torch.cuda.FloatTensor
        else:
            self.tensor = torch.Tensor

        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.max_sequence_length = max_sequence_length
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.device = device
        
        self.embedding_p = nn.Embedding(
                            num_embeddings = self.vocab_size,
                            embedding_dim = self.embedding_size
                            )

        self.discriminator = nn.LSTM(
                        input_size = self.embedding_size,
                        hidden_size = self.hidden_size,
                        num_layers = self.num_layers,
                        bidirectional = self.bidirectional,
                        batch_first = True
                        )

        if self.bidirectional:
            self.output_to_discriminator = nn.Linear(self.max_sequence_length * self.hidden_size * 2, 1)
        else:
            self.output_to_discriminator = nn.Linear(self.max_sequence_length * self.hidden_size, 1)

    def forward(self, paraphrase_sequence):
        paraphrase_sequence = paraphrase_sequence.to(self.device)
        batch_size = paraphrase_sequence.size(0)

        paraphrase_embedding = self.embedding_p(paraphrase_sequence)

        output, (_, _) = self.discriminator(paraphrase_embedding)

        if self.bidirectional:
            output = output.reshape(batch_size, self.max_sequence_length * self.hidden_size * 2)
        else:
            output = output.reshape(batch_size, self.max_sequence_length * self.hidden_size)

        output = self.output_to_discriminator(output)

        logits = torch.sigmoid(output)

        return logits
