
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:03:06 2019

@author: raiv
what: Seq to Seq models that  predict last timestep t and future sequence
 how: concatenates target history y(0->t-1) and x(0->t)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
debugPrint=0;

class CycleEncoderDecoder(nn.Module):
    
    def __init__(self, encoder, decoder,model_params):
        super(CycleEncoderDecoder, self).__init__()
        self.encoder = encoder(model_params)
        self.decoder = decoder(model_params)
        
        self.batch_size = model_params['batch_size']
        
        self.seq_len= model_params['seq_length']
        self.target_len = model_params['target_len']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def train_model(self, input_tensor, target_tensor, n_epochs, target_len, batch_size, training_prediction = 'recursive', teacher_forcing_ratio = 0.5, learning_rate = 0.01, dynamic_tf = False):
        
        '''
        train lstm encoder-decoder
        
        : param input_tensor:              input data with shape (seq_len, # in batch, number features); PyTorch tensor    
        : param target_tensor:             target data with shape (seq_len, # in batch, number features); PyTorch tensor
        : param n_epochs:                  number of epochs 
        : param target_len:                number of values to predict 
        : param batch_size:                number of samples per gradient update
        : param training_prediction:       type of prediction to make during training ('recursive', 'teacher_forcing', or
        :                                  'mixed_teacher_forcing'); default is 'recursive'
        : param teacher_forcing_ratio:     float [0, 1) indicating how much teacher forcing to use when
        :                                  training_prediction = 'teacher_forcing.' For each batch in training, we generate a random
        :                                  number. If the random number is less than teacher_forcing_ratio, we use teacher forcing.
        :                                  Otherwise, we predict recursively. If teacher_forcing_ratio = 1, we train only using
        :                                  teacher forcing.
        : param learning_rate:             float >= 0; learning rate
        : param dynamic_tf:                use dynamic teacher forcing (True/False); dynamic teacher forcing
        :                                  reduces the amount of teacher forcing for each epoch
        : return losses:                   array of loss function for each epoch
        '''
        
        # initialize array of losses 
        losses = np.full(n_epochs, np.nan)

        optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        criterion = nn.MSELoss()

        # calculate number of batch iterations
        n_batches = int(input_tensor.shape[1] / batch_size)

        with trange(n_epochs) as tr:
            for it in tr:
                
                batch_loss = 0.
                batch_loss_tf = 0.
                batch_loss_no_tf = 0.
                num_tf = 0
                num_no_tf = 0

                for b in range(n_batches):
                    # select data 
                    input_batch = input_tensor[:, b: b + batch_size, :]
                    target_batch = target_tensor[:, b: b + batch_size, :]

                    # outputs tensor
                    outputs = torch.zeros(target_len, batch_size, input_batch.shape[2])

                    # initialize hidden state
                    encoder_hidden = self.encoder.init_hidden(batch_size)

                    # zero the gradient
                    optimizer.zero_grad()

                    # encoder outputs
                    encoder_output, encoder_hidden = self.encoder(input_batch)

                    # decoder with teacher forcing
                    decoder_input = input_batch[-1, :, :]   # shape: (batch_size, input_size)
                    decoder_hidden = encoder_hidden

                    if training_prediction == 'recursive':
                        # predict recursively
                        for t in range(target_len): 
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            decoder_input = decoder_output

                    if training_prediction == 'teacher_forcing':
                        # use teacher forcing
                        if random.random() < teacher_forcing_ratio:
                            for t in range(target_len): 
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = target_batch[t, :, :]

                        # predict recursively 
                        else:
                            for t in range(target_len): 
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = decoder_output

                    if training_prediction == 'mixed_teacher_forcing':
                        # predict using mixed teacher forcing
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            
                            # predict with teacher forcing
                            if random.random() < teacher_forcing_ratio:
                                decoder_input = target_batch[t, :, :]
                            
                            # predict recursively 
                            else:
                                decoder_input = decoder_output

                    # compute the loss 
                    loss = criterion(outputs, target_batch)
                    batch_loss += loss.item()
                    
                    # backpropagation
                    loss.backward()
                    optimizer.step()

                # loss for epoch 
                batch_loss /= n_batches 
                losses[it] = batch_loss

                # dynamic teacher forcing
                if dynamic_tf and teacher_forcing_ratio > 0:
                    teacher_forcing_ratio = teacher_forcing_ratio - 0.02 

                # progress bar 
                tr.set_postfix(loss="{0:.3f}".format(batch_loss))
                    
        return losses

    def predict(self, input_tensor, target_len):
        
        '''
        : param input_tensor:      input data (seq_len, input_size); PyTorch tensor 
        : param target_len:        number of target values to predict 
        : return np_outputs:       np.array containing predicted values; prediction done recursively 
        '''

        # encode input_tensor
        input_tensor = input_tensor.unsqueeze(1)     # add in batch size of 1
        encoder_output, encoder_hidden = self.encoder(input_tensor)

        # initialize tensor for predictions
        outputs = torch.zeros(target_len, input_tensor.shape[2])

        # decode input_tensor
        decoder_input = input_tensor[-1, :, :]
        decoder_hidden = encoder_hidden
        
        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output.squeeze(0)
            decoder_input = decoder_output
            
        np_outputs = outputs.detach().numpy()
        
        return np_outputs

    def setOptimizer(self,enc_opt, dec_opt):
        self.encoder_optimizer=enc_opt
        self.decoder_optimizer=dec_opt
        
    def zeroGrad(self):
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            
    def stepOptimizer(self):
             self.encoder_optimizer.step()
             self.decoder_optimizer.step()  
             
    def set_phase(self,phase):
        if phase =='val' or phase =='test':
            self.eval()  
        elif phase=='train':
            self.train()          
             
class CycleEncoder(nn.Module):

    def __init__(self, model_params):
#         __init__(self, num_features, hidden_dim, batch_size, output_dim=1,
#                    num_layers=2,seq_len=1):
        super(Encoder_Ankle, self).__init__()
        self.num_features =model_params['num_features']
        self.hidden_dim = model_params['num_hid_units']
        self.batch_size = model_params['batch_size']
        self.num_layers = model_params['num_layers']
        self.seq_len= model_params['seq_length']
        self.out_dim=1;
        # Define the LSTM layer
        self.enc_lstm = nn.LSTM(self.num_features, self.hidden_dim, self.num_layers)


    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, x_input):

#        input_seqs=input_seqs.permute(1,0,2) # seq length first, batch, input_features
#        enc_states_t,encoder_last = self.enc_lstm(input_seqs)        
#        
        lstm_out, self.hidden = self.enc_lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))
       
        return lstm_out, self.hidden


class CycleDecoder(nn.Module):
    def __init__(self, model_params):
        super(Decoder_Ankle, self).__init__()
        self.num_features =model_params['num_features']
        self.hidden_dim = model_params['num_hid_units']
        self.batch_size = model_params['batch_size']
        self.num_layers = model_params['num_layers']
        self.target_len = model_params['target_len']
        
        self.dec_lstm = nn.LSTM(self.hidden_dim, self.hidden_dim,self.num_layers)
        
        self.dec_output_FC = nn.Linear(self.hidden_dim, 1) # single step output
#        
#        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
#                            num_layers = num_layers)
#        self.linear = nn.Linear(hidden_size, input_size)           

        #output, (h_n, c_n)
        #output of shape (seq_len, batch, num_directions * hidden_size)
        
    def forward(self, x_input, encoder_hidden_states):
        
        '''        
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence 
 
        '''
        
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))     
        
        return output, self.hidden
    
