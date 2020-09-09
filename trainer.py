#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from torch.utils.data import DataLoader,Dataset, Subset
import numpy as np
import tft_model
from data_formatters import ts_dataset  
import data_formatters.base
import expt_settings.configs
import importlib
from data_formatters import utils
import torch.optim as optim


# In[2]:


pd.set_option('max_columns', 1000)


# In[22]:


importlib.reload(tft_model)
importlib.reload(utils)


# In[4]:


ExperimentConfig = expt_settings.configs.ExperimentConfig

config = ExperimentConfig('electricity', 'outputs')
data_formatter = config.make_data_formatter()


print("*** Training from defined parameters for {} ***".format('electricity'))
data_csv_path = 'data/hourly_electricity.csv'
print("Loading & splitting data...")
raw_data = pd.read_csv(data_csv_path, index_col=0)
raw_data = raw_data[raw_data['categorical_id']=='MT_001']
train, valid, test = data_formatter.split_data(raw_data, valid_boundary=1300, test_boundary=1325)
train_samples, valid_samples = data_formatter.get_num_samples_for_calibration(
)

# Sets up default params
fixed_params = data_formatter.get_experiment_params()
params = data_formatter.get_default_model_params()


# In[5]:


#len(train.id.unique())


# In[6]:
id_col = 'categorical_id'
time_col='hours_from_start'
input_cols =['power_usage', 'hour', 'day_of_week', 'hours_from_start', 'categorical_id']
target_col = 'power_usage'

time_steps=192
num_encoder_steps = 168
output_size = 1
max_samples = 1000
input_size = 5

elect = ts_dataset.TSDataset(id_col, [id_col, id_col], time_col, input_cols,
                      target_col, time_steps, max_samples,
                     input_size, num_encoder_steps, 2, output_size, train)


# In[7]:


batch_size=64
loader = DataLoader(
            elect,
            batch_size=batch_size,
            num_workers=2,
            shuffle=True
        )


# In[8]:


for batch in loader:
    break


# In[9]:


static_cols = ['meter']
categorical_cols = ['hour']
real_cols = ['power_usage','hour', 'day']
config = {}
config['static_variables'] = len(static_cols)
config['time_varying_categoical_variables'] = 1
config['time_varying_real_variables_encoder'] = 4
config['time_varying_real_variables_decoder'] = 3
config['num_masked_series'] = 1
config['static_embedding_vocab_sizes'] = [369]
config['time_varying_embedding_vocab_sizes'] = [369]
config['embedding_dim'] = 8
config['lstm_hidden_dimension'] = 160
config['lstm_layers'] = 1
config['dropout'] = 0.05
config['device'] = 'cpu'
config['batch_size'] = 64
config['encode_length'] = 168
config['attn_heads'] = 4
config['num_quantiles'] = 3
config['vailid_quantiles'] = [0.1,0.5,0.9]


# In[23]:


model = tft_model.TFT(config)


# In[11]:


output,encoder_output, decoder_output, attn,attn_output_weights, static_embedding, embeddings_encoder, embeddings_decoder = model.forward(batch)


# In[12]:


output.shape


# In[24]:


q_loss_func = tft_model.QuantileLoss([0.1,0.5,0.9])


# In[ ]:


optimizer = optim.Adam(model.parameters(), lr=0.0001)
model.train()
epochs=10
losses = []
for i in range(epochs):
    epoch_loss = [] 
    j=0
    for batch in loader:
        output, encoder_ouput, decoder_output, attn, attn_weights, _, _, _ = model(batch)
        loss= q_loss_func(output[:,:,:].view(-1,3), batch['outputs'][:,:,0].flatten().float())
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
        j+=1
        if j>5:
            break
    losses.append(np.mean(epoch_loss))
    print(np.mean(epoch_loss))
    


# In[26]:


output, encoder_ouput, decoder_output, attn, attn_weights, _, _, _ = model(batch)


# In[37]:


import matplotlib.pyplot as plt
import numpy as np

ind = np.random.choice(64)
print(ind)
plt.plot(output[ind,:,0].detach().cpu().numpy(), label='pred_1')
plt.plot(output[ind,:,1].detach().cpu().numpy(), label='pred_5')
plt.plot(output[ind,:,2].detach().cpu().numpy(), label='pred_9')

plt.plot(batch['outputs'][ind,:,0], label='true')
plt.legend()


# In[224]:


attn_weights.shape


# In[225]:


plt.matshow(attn_weights.detach().numpy()[0,:,:])


# In[226]:


plt.imshow(attn_weights.detach().numpy()[0,:,:])


# In[ ]:




