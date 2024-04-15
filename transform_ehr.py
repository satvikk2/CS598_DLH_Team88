# -*- coding: utf-8 -*-
"""DL4H_Team_88
# TransformEHR: transformer-based encoder-decoder generative model to enhance prediction of disease outcomes using electronic health records.
**CS598 Project Draft**

Anikesh Haran - anikesh2@illinois.edu         
Satvik Kulkarni - satvikk2@illinois.edu         
Changhua Zhan - zhan36@illinois.edu
"""
# import  packages you need
import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle

# Path to the saved data object
# Assuming your data object is named 'data_object'
mimic4_ds_seqs_path = 'data/PKL/seqs.pkl'
mimic4_ds_visit_dates_path = 'data/PKL/dates.pkl'
mimic4_ds_type_path = 'data/PKL/type.pkl'
mimic4_ds_gender_path = 'data/PKL/gender.pkl'
mimic4_ds_race_path = 'data/PKL/race.pkl'
mimic4_ds_age_path = 'data/PKL/age.pkl'

# Load the data object from Google Drive
with open(mimic4_ds_seqs_path, 'rb') as f:
    seqs = pickle.load(f)

# Load the data object from Google Drive
with open(mimic4_ds_visit_dates_path, 'rb') as f:
    visit_dates = pickle.load(f)

# Load the data object from Google Drive
with open(mimic4_ds_type_path, 'rb') as f:
    icd_codes_types = pickle.load(f)

# Load the data object from Google Drive
with open(mimic4_ds_gender_path, 'rb') as f:
    gender = pickle.load(f)

# Load the data object from Google Drive
with open(mimic4_ds_race_path, 'rb') as f:
    race = pickle.load(f)

# Load the data object from Google Drive
with open(mimic4_ds_age_path, 'rb') as f:
    age = pickle.load(f)

print(len(seqs))
print(len(visit_dates))
print(len(gender))
print(len(race))
print(len(age))
print(len(icd_codes_types))

from torch.utils.data import Dataset

class CustomDataset(Dataset):
  def __init__(self, seqs, gender, race, age, visit_dates):
    self.x = seqs
    self.gender = gender
    self.race = race
    self.age = age
    self.visit_dates = visit_dates

  def __len__(self):
    # your code here
    return len(self.x)

  def __getitem__(self, index):
    # Extract the sequence
    sequence = self.x[index]
    gender = self.gender[index]
    race = self.race[index]
    age = self.age[index]
    visit_dates = self.visit_dates[index]
    # Return the pair (sequence, hf)
    print(sequence, gender, race, age, visit_dates)
    return (sequence, gender, race, age, visit_dates)

dataset = CustomDataset(seqs, gender, race, age, visit_dates)

print(dataset.__getitem__(0))

"""Now we have CustomDataset and collate_fn(). Let us split the dataset into training and validation sets."""

from torch.utils.data.dataset import random_split

split = int(len(dataset)*0.8)

lengths = [split, len(dataset) - split]
train_dataset, val_dataset = random_split(dataset, lengths)

print("Length of train dataset:", len(train_dataset))
print("Length of val dataset:", len(val_dataset))

from torch.utils.data import DataLoader
import math

def load_data(dataset, batch_size):
  def collate_fn(data):
    """
    TODO: Collate the the list of samples into batches. For each patient, you need to pad the diagnosis
          sequences and masks to the sample shape (max # visits, max # diagnosis codes).
    Arguments:
        data: a list of samples fetched from `CustomDataset`
    Outputs:
        x: a tensor of shape (# patiens, max # visits, max # diagnosis codes) of type torch.long
        masks: a tensor of shape (# patiens, max # visits, max # diagnosis codes) of type torch.bool
        gender: a tensor of shape (# patients) of type torch.long (optional)
        race: a tensor of shape (# patients) of type torch.long (optional)
        visit_dates: a list of lists of strings representing visit dates (optional)
    Note that you can obtains the list of diagnosis codes, gender, race and visit dates using:
        sequences, gender, race, visit_dates = zip(*data)`
    """
    def get_position_encoding(position, d_model):
      """Calculates sinusoidal position encoding for a given position and embedding dimension."""
      pe = torch.zeros(d_model)
      div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
      pe[0::2] = torch.sin(position * div_term)
      pe[1::2] = torch.cos(position * div_term)
      return pe.unsqueeze(0)

    sequences, gender, race, age, visit_dates = zip(*data)
    # Convert gender and race to tensors (optional)
    if gender is not None:
      gender = torch.tensor(gender, dtype=torch.long)
    if race is not None:
      race = torch.tensor(race, dtype=torch.long)
    if age is not None:
      age = torch.tensor(age, dtype=torch.long)

    d_model = 2
    num_patients = len(sequences)
    num_visits = [len(patient) for patient in sequences]
    num_codes = [len(visit) for patient in sequences for visit in patient]
    max_num_visits = max(num_visits)
    max_num_codes = max(num_codes)
    x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.long)
    masks = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.bool)
    position_encodings = torch.zeros((num_patients, max_num_visits, d_model), dtype=torch.float)  # For position encoding

    for i_patient, (patient, visit_date) in enumerate(zip(sequences, visit_dates)):
      for j_visit, visit in enumerate(patient):
        # Mask all ICD codes in the visit
        masked_visit = [0] * len(visit)  # Replace with actual masking logic (e.g., random masking)
        padded_seq = torch.tensor(masked_visit + [0] * (max_num_codes - len(visit)), dtype=torch.long)
        x[i_patient, j_visit, :] = padded_seq
        masks[i_patient, j_visit, :] = torch.ones(max_num_codes, dtype=torch.bool)  # All codes masked in this visit
        # Calculate position encoding based on visit date (assuming YYYY-MM-DD format)
        year, month, day = map(int, visit_date[j_visit].split('-'))
        # You can customize the date processing logic based on your data format
        date_as_float = year + (month - 1) / 12 + day / (365.25 * 12)  # Approximate date as float
        position_encodings[i_patient, j_visit, :] = get_position_encoding(date_as_float, d_model)

    return (x, masks, gender, race, age, position_encodings)

  return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

train_loader = load_data(train_dataset, batch_size = 32)
val_loader = load_data(val_dataset,  batch_size = 32)

#Check the loader and collate function implementation
loader_iter = iter(val_loader)
x, masks, gender, race, age, position_encodings = next(loader_iter)
print(x, masks, gender, race, age, position_encodings)

print("x", x.shape)
print("masks", masks.shape)
print("gender", gender.shape)
print("race", race.shape)
print("age", age.shape)
print("position_encodings", position_encodings.shape)

# Define the number of classes for each categorical feature
num_gender_classes = 2
num_race_classes = 33
# Define the maximum number of visits and diagnosis codes
max_num_visits = 18
max_num_codes = 35

class TranformEHR(nn.Module):
  def __init__(self, num_gender_classes, num_race_classes, num_code, nhead, num_encoder_layers, num_decoder_layers, embedding_dim=128):
    super().__init__()
    # Define the embedding layers
    self.gender_embedding = nn.Embedding(num_gender_classes, embedding_dim)
    self.race_embedding = nn.Embedding(num_race_classes, embedding_dim)
    # Define the embeddings for other continuous features (age, position_encodings)
    self.age_embedding = nn.Linear(1, embedding_dim)  # Assuming age is a continuous feature
    self.position_encodings_embedding = nn.Linear(2, embedding_dim)  # Assuming position_encodings has 2 dimensions
    self.visit_embedding = nn.Embedding(num_embeddings=num_code, embedding_dim=embedding_dim)
    # Transformer encoder
    encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
    # Transformer decoder
    decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=nhead)
    self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
    # Linear layer to project decoder output to ICD code probabilities
    self.linear = nn.Linear(embedding_dim, num_code)

  def forward(self, x, masks, gender, race, age, position_encodings):
    embedded_gender = self.gender_embedding(gender)
    embedded_race = self.race_embedding(race)
    #embedded_age = self.age_embedding(age.float())
    embedded_age = self.age_embedding(age.view(-1, 1).float())
    embedded_positional_encodings = self.position_encodings_embedding(position_encodings)
    embedded_x = self.visit_embedding(x)

    # Concatenate all embeddings
    print(embedded_x.shape)
    print(embedded_positional_encodings.shape)
    print(embedded_age.shape)
    print(embedded_race.shape)
    print(embedded_gender.shape)
    # Add dimensions to positional encodings and other embeddings
    embedded_positional_encodings = embedded_positional_encodings.unsqueeze(2).expand(-1, -1, 27, -1)
    embedded_age = embedded_age.unsqueeze(1).unsqueeze(1).expand(-1, 13, 27, -1)
    embedded_race = embedded_race.unsqueeze(1).unsqueeze(1).expand(-1, 13, 27, -1)
    embedded_gender = embedded_gender.unsqueeze(1).unsqueeze(1).expand(-1, 13, 27, -1)
    embedded_input = torch.cat((embedded_x, embedded_positional_encodings, embedded_age, embedded_race, embedded_gender), dim=2)
    print(embedded_input.shape)
    reshaped_input = embedded_input.reshape(32, 13*27, -1)
    print(reshaped_input.shape)
    # Apply transformer encoder
    print(masks.shape)
    encoder_output = self.transformer_encoder(reshaped_input, src_key_padding_mask=masks.reshape(32, 13*27))

    # Apply transformer decoder
    decoder_output = self.transformer_decoder(embedded_input, encoder_output, tgt_key_padding_mask=masks)

    # Project decoder output to ICD code probabilities
    logits = self.linear(decoder_output)

    return logits

# load the model here
model = TranformEHR(num_gender_classes, num_race_classes, num_code=len(icd_codes_types), nhead=2, num_encoder_layers=1, num_decoder_layers=1)

# HZ version

# Define the number of classes for each categorical feature
num_gender_classes = 2
num_race_classes = 33
# Define the maximum number of visits and diagnosis codes
max_num_visits = 18
max_num_codes = 35

class TranformEHR(nn.Module):
  def __init__(self, num_gender_classes, num_race_classes, num_code, nhead, num_encoder_layers, num_decoder_layers, embedding_dim=128):
    super().__init__()
    # HZ
    self.embedding_dim = embedding_dim
    self.concatenated_dim = embedding_dim * 5
    self.projected_dim = embedding_dim


    # Define the embedding layers
    self.gender_embedding = nn.Embedding(num_gender_classes, embedding_dim)
    self.race_embedding = nn.Embedding(num_race_classes, embedding_dim)
    # Define the embeddings for other continuous features (age, position_encodings)
    self.age_embedding = nn.Linear(1, embedding_dim)  # Assuming age is a continuous feature
    self.position_encodings_embedding = nn.Linear(2, embedding_dim)  # Assuming position_encodings has 2 dimensions
    self.visit_embedding = nn.Embedding(num_embeddings=num_code, embedding_dim=embedding_dim)

    # HZ
    # self.embedding_projection = nn.Linear(embedding_dim * 5, self.projected_dim)
    self.embedding_projection = nn.Linear(self.concatenated_dim, self.projected_dim)

    # Transformer encoder
    encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
    # Transformer decoder
    # decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=nhead)
    decoder_layer = nn.TransformerDecoderLayer(d_model=self.projected_dim, nhead=nhead)

    self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
    # Linear layer to project decoder output to ICD code probabilities
    # self.linear = nn.Linear(embedding_dim, num_code)
    self.linear = nn.Linear(self.projected_dim, num_code)

  def forward(self, x, masks, gender, race, age, position_encodings):
    embedded_gender = self.gender_embedding(gender)
    embedded_race = self.race_embedding(race)
    #embedded_age = self.age_embedding(age.float())
    # HZ
    # embedded_age = self.age_embedding(age.view(-1, 1).float())
    embedded_age = self.age_embedding(age.float().unsqueeze(-1))
    embedded_positional_encodings = self.position_encodings_embedding(position_encodings.float())
    embedded_x = self.visit_embedding(x)

    # Concatenate all embeddings
    print(embedded_x.shape)
    print(embedded_positional_encodings.shape)
    print(embedded_age.shape)
    print(embedded_race.shape)
    print(embedded_gender.shape)
    # HZ
    # Add dimensions to positional encodings and other embeddings
    # embedded_positional_encodings = embedded_positional_encodings.unsqueeze(2).expand(-1, -1, 27, -1)
    # embedded_age = embedded_age.unsqueeze(1).unsqueeze(1).expand(-1, 13, 27, -1)
    # embedded_race = embedded_race.unsqueeze(1).unsqueeze(1).expand(-1, 13, 27, -1)
    # embedded_gender = embedded_gender.unsqueeze(1).unsqueeze(1).expand(-1, 13, 27, -1)
    embedded_positional_encodings = embedded_positional_encodings.unsqueeze(2).expand(-1, -1, embedded_x.size(2), -1)
    embedded_age = embedded_age.unsqueeze(1).unsqueeze(2).expand(-1, embedded_x.size(1), embedded_x.size(2), -1)
    embedded_race = embedded_race.unsqueeze(1).unsqueeze(2).expand(-1, embedded_x.size(1), embedded_x.size(2), -1)
    embedded_gender = embedded_gender.unsqueeze(1).unsqueeze(2).expand(-1, embedded_x.size(1), embedded_x.size(2), -1)

    # print("Shape of embedded_x:", embedded_x.shape)
    # print("Shape of embedded_positional_encodings:", embedded_positional_encodings.shape)
    # print("Shape of embedded_age:", embedded_age.shape)
    # print("Shape of embedded_race:", embedded_race.shape)
    # print("Shape of embedded_gender:", embedded_gender.shape)
    embedded_input = torch.cat((embedded_x, embedded_positional_encodings, embedded_age, embedded_race, embedded_gender), dim=-1)
    # HZ
    embedded_input = self.embedding_projection(embedded_input)

    print(embedded_input.shape)
    # HZ
    # reshaped_input = embedded_input.reshape(32, 13*27, -1)
    reshaped_input = embedded_input.reshape(embedded_input.size(0), -1, self.projected_dim)
    print(reshaped_input.shape)
    # Apply transformer encoder
    print("shape of masks: ", masks.shape)
    # HZ
    # reshaped_masks = masks.reshape(masks.size(0), -1)
    if masks is not None:
        reshaped_masks = masks.view(masks.size(0), -1)

    # Confirm sizes match
    print(f"Input size: {embedded_input.shape}")  # Expected [batch_size, seq_len, features]
    print(f"Mask size: {reshaped_masks.shape}")   # Expected [batch_size, seq_len]

    # encoder_output = self.transformer_encoder(reshaped_input, src_key_padding_mask=masks.reshape(32, 13*27))
    encoder_output = self.transformer_encoder(reshaped_input, src_key_padding_mask= reshaped_masks)

    # Apply transformer decoder
    decoder_output = self.transformer_decoder(embedded_input, encoder_output, tgt_key_padding_mask=masks)

    # Project decoder output to ICD code probabilities
    logits = self.linear(decoder_output)

    return logits

# load the model here
model = TranformEHR(num_gender_classes, num_race_classes, num_code=len(icd_codes_types), nhead=2, num_encoder_layers=1, num_decoder_layers=1)

import torch
from torch import nn
from torch.utils.data import DataLoader

# Define your loss function (e.g., cross-entropy)
criterion = nn.CrossEntropyLoss()

# Define your optimizer (e.g., Adam)
optimizer = torch.optim.Adam(model.parameters())

log_interval = 5

def train(model, train_data_loader, epochs):
  """
  Train the TransformEHR model on the provided dataloader.

  Args:
      model (nn.Module): The TransformEHR model to train.
      dataloader (DataLoader): The dataloader containing training data.
      epochs (int): Number of training epochs.
  """
  model.train()  # Set model to training mode
  for epoch in range(epochs):
    for batch_idx, batch in enumerate(train_loader):
      x, masks, gender, race, age, position_encodings = batch
      # Move data to the device (GPU if available)
      #x, masks, gender, race, age, position_encodings, labels = x.to(device), masks.to(device), gender.to(device), race.to(device), age.to(device), position_encodings.to(device), labels.to(device)
      # Forward pass
      logits = model(x, masks, gender, race, age, position_encodings)
      loss = criterion(logits.view(-1, logits.size(-1)), x.view(-1))

      # Backward pass and update parameters
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # Print training progress (optional)
      if (i + 1) % log_interval == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_data_loader)}], Loss: {loss.item():.4f}")

def eval(model, val_data_loader):
  """
  Evaluate the TransformEHR model on the provided dataloader.

  Args:
      model (nn.Module): The TransformEHR model to evaluate.
      dataloader (DataLoader): The dataloader containing evaluation data.

  Returns:
      float: Average loss on the evaluation data
  """
  model.eval()  # Set model to evaluation mode
  with torch.no_grad():  # Disable gradient calculation for efficiency
    total_loss = 0
    for x, masks, gender, race, age, position_encodings in val_data_loader:
      # Move data to the device (GPU if available)
      #x, masks, gender, race, age, position_encodings, labels = x.to(device), masks.to(device), gender.to(device), race.to(device), age.to(device), position_encodings.to(device), labels.to(device)

      # Forward pass
      logits = model(x, masks, gender, race, age, position_encodings)
      loss = criterion(logits.view(-1, logits.size(-1)), x.view(-1))
      total_loss += loss.item()

    # Calculate average loss
    avg_loss = total_loss / len(val_data_loader)
    return avg_loss

# Example usage
train(model, train_loader, 2)
eval_loss = eval(model, val_loader)
print(f"Evaluation Loss: {eval_loss:.4f}")