# -*- coding: utf-8 -*-
#TransformEHR: transformer-based encoder-decoder generative model to enhance prediction of disease outcomes using electronic health records.

# import  packages
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import math
import torch.nn.utils.rnn as rnn_utils

import pandas as pd

# Define the data for the table
common_outcomes = {
    'ICD-10-CM Code': ['I10', 'E785', 'Z87891', 'K219', 'F329', 'I2510', 'F419', 'N179', 'Z794', 'Z7901'],
    'Description': [
        'Essential (primary) hypertension',
        'Hyperlipidemia, unspecified',
        'Personal history of nicotine dependence',
        'Gastro-esophageal reflux disease without esophagitis',
        'Major depressive disorder, unspecified',
        'Atherosclerotic heart disease of native coronary artery without angina pectoris',
        'Unspecified anxiety disorder',
        'Chronic kidney disease, unspecified',
        'Long-term (current) use of insulin',
        'Long-term (current) use of opiate analgesic'
    ]
}

# Create a DataFrame from the data
common_outcomes_df = pd.DataFrame(common_outcomes)

# Display the DataFrame
common_outcomes_df

"""**Table 2 - Uncommon ICD-10CM Codes**"""

import pandas as pd

# Define the data for the table
uncommon_outcomes = {
    'ICD-10-CM Code': ['N94.6', 'T47.1X5D', 'O30.033', 'I70234', 'I95.2', 'Z34.83', 'C8518', 'L89.891', 'D126', 'I201'],
    'Description': [
        'Dyspareunia, unspecified',
        'Poisoning by antineoplastic and immunosuppressive drugs, accidental (unintentional), subsequent encounter',
        'Triplet pregnancy, fetus 3',
        'Atherosclerosis of native arteries of extremities with gangrene, bilateral legs',
        'Hypotension, unspecified',
        'Supervision of high-risk pregnancy with other poor reproductive or obstetric history',
        'Diffuse large B-cell lymphoma, lymph nodes of axilla and upper limb',
        'Pressure ulcer of other site, stage 1',
        'Benign neoplasm of colon',
        'Unstable angina'
    ]
}

# Create a DataFrame from the additional data
uncommon_outcomes_df = pd.DataFrame(uncommon_outcomes)

# Display the DataFrame
uncommon_outcomes_df

"""**Patient Age - PreProcessing**

Current PyHealth based data processing does not compute age feature. hence we pre-processed the patient's age separately and created a pickle files for age feature for quick loading during model training.

import csv

# Define the path to the CSV file
root = '/content/drive/MyDrive/DLH/MIMIC4/CSV/'
patient_file_path = root + 'patients.csv'

id2age = {}

# read id and age from patients.csv and save it in a dictionary id2age
def read_patient_age(file_path):
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            id2age[row[0]] = int(row[2])

read_patient_age(patient_file_path)"

**Pre-Processing - transform_ehr_mimic4_fn**

We have developed function **transform_ehr_mimic4_fn** to process individual patients and create feautres such as visit level details, icd codes and patinet;s demographic details such as age, gender and race.

To reduce the data complexity and need of high compute power, we have pre-processed the longitudnal EHR data and kept fixed length sequence of Visit & ICD-Codes.

* Visit Length - 4 visits per patient
* ICD Codes - 5 ICD codes per visit

Patients with less then 4 visits and less then 5 ICD diagnosis-codes have been discarded from the pre-traning cohort.

**Total Patients In Pre-Training Cohort - 23206**

# Compute sequenced data for learning embeddings

from datetime import datetime
from pyhealth.medcode import CrossMap
import random
# set the random seed
random.seed(0)

# load the mapping from ICD9CM to CCSCM
mapping_icd9cm_ccscm = CrossMap.load(source_vocabulary="ICD9CM", target_vocabulary="CCSCM")
# load the mapping from CCSCM to ICD10CM
mapping_ccscm_icd10cm = CrossMap.load(source_vocabulary="CCSCM", target_vocabulary="ICD10CM")

#Calculate Patient's Age
def calculate_age(birth_date, death_date):
  # Calculate age
  age = death_date.year - birth_date.year - ((death_date.month, death_date.day) < (birth_date.month, birth_date.day))
  return age

types = {}
gender2idx = {}
race2idx = {}

def transform_ehr_mimic4_fn(patient):
    visit_idx = []
    newPatient = []
    age = 0
    gender = 0
    race = 0
    visit_dates = []
    #consider patient with 4 or more visits
    keep_patient = True
    if len(patient) >= 4:
      for i in range(len(patient)):
        #visit level details
        visit_idx.append(1 + i)

        visit = patient[i]
        conditions = []
        events = visit.get_event_list(table="diagnoses_icd")
        if(len(events) < 5):
          continue
        formatted_visit_date = visit.encounter_time.strftime("%Y-%m-%d")
        visit_dates.append(formatted_visit_date)

        for event in events:
          vocabulary = event.vocabulary
          code = ""
          if vocabulary == "ICD9CM":
            # map from ICD9CM to CCSCM
            ccscmCodes = mapping_icd9cm_ccscm.map(event.code)
            # in the case where one ICD9CM code maps to multiple CCSCM codes, randomly select one
            ccscmCode = random.choice(ccscmCodes)

            # map from CCSCM to ICD10CM
            icd10cmCodes = mapping_ccscm_icd10cm.map(ccscmCode)
            # in the case where one CCSCM code maps to multiple ICD10CM codes, randomly select one
            code = random.choice(icd10cmCodes)
          else:
            code = event.code

          if code in types:
            conditions.append(types[code])
          else:
            types[code] = len(types)
            conditions.append(types[code])

        # step 2: assemble the sample
        # if conditions is not empty, add the sample
        # if (conditions): # commented it out because len(visit_date) needs to be the same as len(newPatient)
        newPatient.append(conditions)

      if(len(conditions) >= 4):
        #visits.append(visit_idx)
        if len(newPatient) > 100:
          print(patient.patient_id,)
        #visit_dates.append(visit_date)
        #age.append(patient.anchor_age)

        # get age of patient using patient id and id2age dictionary
        #age.append(id2age[patient.patient_id])
        age = id2age[patient.patient_id]

        p_gender = patient.gender
        if p_gender in gender2idx:
          #gender.append(gender2idx[p_gender])
          gender = gender2idx[p_gender]
        else:
          gender2idx[p_gender] = len(gender2idx)
          #gender.append(gender2idx[p_gender])
          gender = gender2idx[p_gender]

        p_ethnicity = patient.ethnicity
        if p_ethnicity in race2idx:
          #race.append(race2idx[p_ethnicity])
          race = race2idx[p_ethnicity]
        else:
          race2idx[p_ethnicity] = len(race2idx)
          #race.append(race2idx[p_ethnicity])
          race = race2idx[p_ethnicity]
    return newPatient, visit_idx, age, gender, race, visit_dates
"""

"""# Pre-processing of MIMIC4 Data
seqs = []
all_visits = []
all_age = []
all_gender = []
all_race = []
all_visit_dates = []
patient_dict = mimic4_data.patients
for patient_id in mimic4_data.patients:
  patient = patient_dict[patient_id]
  seq, visit_numbers, age, gender, race, visit_dates = transform_ehr_mimic4_fn(patient)
  if seq and len(seq) >=4:
    seqs.append(seq)
    all_visits.append(visit_numbers)
    all_age.append(age)
    all_gender.append(gender)
    all_race.append(race)
    all_visit_dates.append(visit_dates)

print(seqs[0])"""
#[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 20, 4, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 14, 31, 32, 4, 33, 34]]

"""**Ordering Visits Based on Visit Dates**"""

"""# sort the visit_date and seqs based on the visit date
from datetime import datetime
sorted_seqs = []
sorted_visit_dates = []
for i in range(len(seqs)):
    visit_date = all_visit_dates[i]
    seq = seqs[i]
    visit_date_seq_tuple = [(visit_date[j], seq[j]) for j in range(len(seq))]
    visit_date_seq_tuple.sort(key=lambda x: datetime.strptime(x[0], "%Y-%m-%d"))

    sorted_visit_dates.append([x[0] for x in visit_date_seq_tuple])
    sorted_seqs.append([x[1] for x in visit_date_seq_tuple])

seqs = sorted_seqs
all_visit_dates = sorted_visit_dates
print(seqs[0])
print(all_visit_dates[0])"""

#[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [24, 25, 26, 27, 28, 29, 30, 14, 31, 32, 4, 33, 34], [16, 17, 18, 19, 20, 20, 4, 21, 22, 23]]
#['2180-05-06', '2180-06-26', '2180-07-23', '2180-08-05']

"""print(len(seqs))
print(len(all_visits))
print(len(all_visit_dates))
print(len(all_gender))
print(len(all_race))
print(len(all_age))
print(len(types))"""

#23206
#23206
#23206
#23206
#23206
#23206
#51730

"""**Create Pickle**

In below code section we have created pickle files for all the data features - sequences, visit dates, gender , race & age and stored into filesystem.

Note - change the "path" according to your environment.

import pickle

mimic4_ds_seqs_path = '/content/seqs.pkl'
mimic4_ds_visits_path = '/content/visits.pkl'
mimic4_ds_visit_dates_path = '/content/dates.pkl'
mimic4_ds_type_path = '/content/type.pkl'
mimic4_ds_gender_path = '/content/gender.pkl'
mimic4_ds_race_path = '/content/race.pkl'
mimic4_ds_age_path = '/content/age.pkl'

# Save the data object to Google Drive
with open(mimic4_ds_seqs_path, 'wb') as f:
    pickle.dump(seqs, f)

with open(mimic4_ds_visits_path, 'wb') as f:
    pickle.dump(all_visits, f)

with open(mimic4_ds_visit_dates_path, 'wb') as f:
    pickle.dump(all_visit_dates, f)

with open(mimic4_ds_type_path, 'wb') as f:
    pickle.dump(types, f)

with open(mimic4_ds_gender_path, 'wb') as f:
    pickle.dump(all_gender, f)

with open(mimic4_ds_race_path, 'wb') as f:
    pickle.dump(all_race, f)

with open(mimic4_ds_age_path, 'wb') as f:
    pickle.dump(all_age, f)

**Load Data for Model Training**
"""

#Load MIMIC4 data from google drive
import pickle

# Path to the saved data object
mimic4_ds_seqs_path = 'data/v3/seqs.pkl'
mimic4_ds_visits_path = 'data/v3/visits.pkl'
mimic4_ds_visit_dates_path = 'data/v3/dates.pkl'
mimic4_ds_type_path = 'data/v3/type.pkl'
mimic4_ds_gender_path = 'data/v3/gender.pkl'
mimic4_ds_race_path = 'data/v3/race.pkl'
mimic4_ds_age_path = 'data/v3/age.pkl'

# Load the data object from Google Drive
with open(mimic4_ds_seqs_path, 'rb') as f:
    seqs = pickle.load(f)

with open(mimic4_ds_visits_path, 'rb') as f:
    visits = pickle.load(f)

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

"""**Build The Dataset**

First, we have implemented a custom dataset using PyTorch Dataset class, which will characterize the key features of the dataset we want to generate.

We will use the sequences of diagnosis-codes, gender, age, race and visit-dates as input for pretraning.
"""

from torch.utils.data import Dataset

class CustomDataset(Dataset):
  def __init__(self, seqs, visits, gender, race, age, visit_dates):
    self.x = seqs
    self.visit = visits
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
    visits = self.visit[index]
    gender = self.gender[index]
    race = self.race[index]
    age = self.age[index]
    visit_dates = self.visit_dates[index]
    # Return the pair (sequence, hf)
    return (sequence, visits, gender, race, age, visit_dates)

dataset = CustomDataset(seqs, visits, gender, race, age, visit_dates)

print(dataset.__getitem__(0))

#Output
#([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [24, 25, 26, 27, 28, 29, 30, 14, 31, 32, 4, 33, 34], [16, 17, 18, 19, 20, 20, 4, 21, 22, 23]], [1, 2, 3, 4], 0, 0, 52, ['2180-05-06', '2180-06-26', '2180-07-23', '2180-08-05'])

"""**Data Sampler & Split Data Into Train and Validation Set**

We have also created a data sampler to quickly sample the data to test the model training, shapes and evaluation steps.
"""

#Run on sample
from torch.utils.data import Dataset, SubsetRandomSampler

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Define the size of the subset (20% of the dataset)
sample_size = float(sys.argv[1])
#sample_size = 0.03 #Sampling only 20 % of the dataset for model tranining and validation
#sample_size = 1.0 #Sampling all of the dataset 100% for model tranining and validation

subset_size = int(sample_size * len(dataset))

# Create a random sampler to sample indices from the dataset
indices = list(range(len(dataset)))
np.random.shuffle(indices)  # Shuffle the indices randomly
subset_indices = indices[:subset_size]  # Take the first subset_size indices

# Create a SubsetRandomSampler using the subset indices
subset_sampler = SubsetRandomSampler(subset_indices)

from torch.utils.data.dataset import random_split

#use subset data and split in 80/20 for train and vel
# Split the subset indices into training and validation indices (80/20 split)
split_index = int(0.8 * len(subset_indices))
train_indices = subset_indices[:split_index]
val_indices = subset_indices[split_index:]

# Create SubsetRandomSamplers for training and validation sets
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

print("Length of train dataset:", len(train_sampler))
print("Length of val dataset:", len(val_sampler))

#Length of train dataset: 18564
#Length of val dataset: 4642

"""**Split Data Into Train and Validation Set**

Another utility to split the dataset into training and validation sets without sampling.
"""

"""from torch.utils.data.dataset import random_split

split = int(len(dataset)*0.8)

lengths = [split, len(dataset) - split]
train_dataset, val_dataset = random_split(dataset, lengths)

print("Length of train dataset:", len(train_dataset))
print("Length of val dataset:", len(val_dataset))"""

#Length of train dataset: 18564
#Length of val dataset: 4642

"""**Data Loader & collate_fn Implementation**

Within collate_fu we are computing positional encoding to embed the time, we applied sinusoidal position embedding [2] to the numerical format of visit date (date-specific)

**Sample Data Loader and Collate Function**
"""

from torch.utils.data import DataLoader
import math
import torch.nn.utils.rnn as rnn_utils

def load_sample_data(dataset, sampler, batch_size, shuffle):
    def collate_fn(data):
        def get_position_encoding(position, d_model):
            """Calculates sinusoidal position encoding for a given position and embedding dimension."""
            pe = torch.zeros(d_model)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            #print("position", position)
            #print("div_term",div_term)
            position *= 2 * math.pi
            pe[0::2] = torch.sin(position * div_term)
            pe[1::2] = torch.cos(position * div_term)
            return pe.unsqueeze(0)

        sequences, visits_ids, gender, race, age, visit_dates = zip(*data)
        # Convert gender and race to tensors (optional)
        if gender is not None:
            gender = torch.tensor(gender, dtype=torch.long)
        if race is not None:
            race = torch.tensor(race, dtype=torch.long)
        if age is not None:
            age = torch.tensor(age, dtype=torch.long)

        sequences = [patient[-4:] for patient in sequences]
        visit_dates = [visit_date[-4:] for visit_date in visit_dates]
        visits_ids = [visit_id[:4] for visit_id in visits_ids]

        #positional encoding dim
        d_model = 2

        num_patients = len(sequences)
        num_visits = [len(patient) for patient in sequences]
        num_codes = [len(visit) for patient in sequences for visit in patient]
        max_num_visits = max(num_visits)
        #max_num_visits = 4
        max_num_codes = 5
        pad_value = 0

        visit_numbers = rnn_utils.pad_sequence([torch.tensor(visit) for visit in visits_ids], batch_first=True,padding_value=0)
        num_heads = 1
        x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.long)
        masks = torch.zeros((num_patients, num_heads, max_num_visits, max_num_codes), dtype=torch.bool)
        attn_masks = torch.zeros((num_patients, max_num_visits, max_num_visits), dtype=torch.bool)
        position_encodings = torch.zeros((num_patients, max_num_visits, d_model),dtype=torch.float)  # For position encoding

        for i_patient, (patient, visit_date) in enumerate(zip(sequences, visit_dates)):
            valid_visits = [visit for visit in patient if len(visit) > 4]
            if len(valid_visits) >= max_num_visits:
                for h in range(num_heads):
                  for j_visit, visit in enumerate(valid_visits[:max_num_visits]):

                      last_5_icd_codes = visit[-5:]

                      x[i_patient, j_visit, :] = torch.tensor(last_5_icd_codes, dtype=torch.long)

                      # Calculate the attention mask
                      attn_mask_row = [1] * (j_visit + 1) + [0] * (max_num_visits - j_visit - 1)
                      attn_masks[i_patient, j_visit] = torch.tensor(attn_mask_row, dtype=torch.bool)

                      # Create mask for the visit (mask all ICD codes in the visit)
                      masks[i_patient, h, j_visit, :len(last_5_icd_codes)] = True

                      if j_visit == len(valid_visits)-1:  # Check if it's the last visit for the patient
                          masks[i_patient, h, j_visit, :] = False

                      # Calculate position encoding based on visit date (assuming YYYY-MM-DD format)
                      year, month, day = map(int, visit_date[j_visit].split('-'))
                      # You can customize the date processing logic based on your data format
                      date_as_float = year + (month - 1) / 12 + day / (365.25 * 12)  # Approximate date as float
                      position_encodings[i_patient, j_visit, :] = get_position_encoding(date_as_float, d_model)

        return (x, attn_masks, masks, visit_numbers, gender, race, age, position_encodings)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, sampler=sampler)

train_loader = load_sample_data(dataset, train_sampler, batch_size=32, shuffle=True)
val_loader = load_sample_data(dataset, val_sampler, batch_size=32, shuffle=False)

"""**All Data Loader & Collate Function**"""

from torch.utils.data import DataLoader
import math
import torch.nn.utils.rnn as rnn_utils

def load_data(dataset, batch_size, shuffle):
    def collate_fn(data):
        def get_position_encoding(position, d_model):
            """Calculates sinusoidal position encoding for a given position and embedding dimension."""
            pe = torch.zeros(d_model)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            #print("position", position)
            #print("div_term",div_term)
            position *= 2 * math.pi
            pe[0::2] = torch.sin(position * div_term)
            pe[1::2] = torch.cos(position * div_term)
            return pe.unsqueeze(0)

        sequences, visits_ids, gender, race, age, visit_dates = zip(*data)
        # Convert gender and race to tensors (optional)
        if gender is not None:
            gender = torch.tensor(gender, dtype=torch.long)
        if race is not None:
            race = torch.tensor(race, dtype=torch.long)
        if age is not None:
            age = torch.tensor(age, dtype=torch.long)

        sequences = [patient[-4:] for patient in sequences]
        visit_dates = [visit_date[-4:] for visit_date in visit_dates]
        visits_ids = [visit_id[:4] for visit_id in visits_ids]

        #positional encoding dim
        d_model = 2
        num_patients = len(sequences)
        num_visits = [len(patient) for patient in sequences]
        num_codes = [len(visit) for patient in sequences for visit in patient]
        max_num_visits = max(num_visits)
        #max_num_visits = 4
        max_num_codes = 5
        pad_value = 0

        visit_numbers = rnn_utils.pad_sequence([torch.tensor(visit) for visit in visits_ids], batch_first=True,padding_value=0)
        num_heads = 1
        x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.long)
        masks = torch.zeros((num_patients, num_heads, max_num_visits, max_num_codes), dtype=torch.bool)
        attn_masks = torch.zeros((num_patients, max_num_visits, max_num_visits), dtype=torch.bool)
        position_encodings = torch.zeros((num_patients, max_num_visits, d_model),dtype=torch.float)  # For position encoding

        for i_patient, (patient, visit_date) in enumerate(zip(sequences, visit_dates)):
            valid_visits = [visit for visit in patient if len(visit) > 4]
            #print(valid_visits)
            if len(valid_visits) >= max_num_visits:
                for h in range(num_heads):
                  for j_visit, visit in enumerate(valid_visits[:max_num_visits]):
                      last_5_icd_codes = visit[-5:]

                      x[i_patient, j_visit, :] = torch.tensor(last_5_icd_codes, dtype=torch.long)

                      attn_mask_row = [1] * (j_visit + 1) + [0] * (max_num_visits - j_visit - 1)
                      attn_masks[i_patient, j_visit] = torch.tensor(attn_mask_row, dtype=torch.bool)

                      masks[i_patient, h, j_visit, :len(last_5_icd_codes)] = True

                      if j_visit == len(valid_visits)-1:  # Check if it's the last visit for the patient
                          masks[i_patient, h, j_visit, :] = False

                      # Calculate position encoding based on visit date (assuming YYYY-MM-DD format)
                      year, month, day = map(int, visit_date[j_visit].split('-'))
                      date_as_float = year + (month - 1) / 12 + day / (365.25 * 12)  # Approximate date as float
                      position_encodings[i_patient, j_visit, :] = get_position_encoding(date_as_float, d_model)

        return (x, attn_masks, masks, visit_numbers, gender, race, age, position_encodings)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

#train_loader = load_data(train_dataset, batch_size = 32)
#val_loader = load_data(val_dataset,  batch_size = 32)

#Check the loader and collate function implementation
loader_iter = iter(train_loader)
x, attn_masks, masks, visit_numbers, gender, race, age, position_encodings = next(loader_iter)
print(x, attn_masks , masks,  visit_numbers, gender, race, age, position_encodings)

#Check shapes for all the features
print("x", x.shape)
print("masks", masks.shape)
print("visits", visit_numbers.shape)
print("gender", gender.shape)
print("race", race.shape)
print("age", age.shape)
print("position_encodings", position_encodings.shape)

# Define the number of classes for each categorical feature
num_gender_classes = 2
num_race_classes = 33
# Define the maximum number of visits and diagnosis codes
#num_visits = [len(visit) for visit in visits]
#max_num_visits = max(num_visits)

max_num_visits = 4
max_num_codes = 5

def get_encoder_mask(batch_size, seq_length):
    # Create a square matrix with ones in the lower triangle (including the diagonal)
    mask = torch.tril(torch.ones(seq_length, seq_length))
    # Expand to match the batch size
    mask = mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, seq_length, seq_length)
    return mask

class TranformEHR_M1(nn.Module):
    def __init__(self, num_gender_classes, num_race_classes, num_visits, num_code, nhead, num_encoder_layers,
                 num_decoder_layers, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.concatenated_dim = embedding_dim * 1
        self.projected_dim = embedding_dim
        self.num_heads = nhead

        # Define the embedding layers
        self.visit_number_embedding = nn.Embedding(num_embeddings=num_visits, embedding_dim=embedding_dim)
        self.gender_embedding = nn.Embedding(num_gender_classes, embedding_dim)
        self.race_embedding = nn.Embedding(num_race_classes, embedding_dim)

        # Define the embeddings for other continuous features (age, position_encodings)
        self.age_embedding = nn.Linear(1, embedding_dim)  #age is a continuous feature
        self.position_encodings_embedding = nn.Linear(2, embedding_dim)  # position_encodings has 2 dimensions

        self.visit_embedding = nn.Embedding(num_embeddings=num_code, embedding_dim=embedding_dim)

        self.embedding_projection = nn.Linear(self.concatenated_dim, self.projected_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True, norm_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.projected_dim, nhead=nhead, batch_first=True, norm_first=True)

        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        # Linear layer to project decoder output to ICD code probabilities
        self.linear = nn.Linear(self.projected_dim, num_code)


    def forward(self, x, attn_masks, masks, visit_numbers, gender, race, age, position_encodings):

        max_visit_number = self.visit_number_embedding.num_embeddings - 1  # Get the max allowed index
        visit_numbers = torch.clamp(visit_numbers, 0, max_visit_number)  # Clamp values to valid range
        embedded_visits_number = self.visit_number_embedding(visit_numbers)
        embedded_gender = self.gender_embedding(gender)
        embedded_race = self.race_embedding(race)
        embedded_age = self.age_embedding(age.float().unsqueeze(-1))
        embedded_positional_encodings = self.position_encodings_embedding(position_encodings.float())

        embedded_x = self.visit_embedding(x)

        """embedded_positional_encodings = embedded_positional_encodings.unsqueeze(2).expand(-1, -1, embedded_x.size(2),-1)
        embedded_age = embedded_age.unsqueeze(1).unsqueeze(2).expand(-1, embedded_x.size(1), embedded_x.size(2), -1)
        embedded_race = embedded_race.unsqueeze(1).unsqueeze(2).expand(-1, embedded_x.size(1), embedded_x.size(2), -1)
        embedded_gender = embedded_gender.unsqueeze(1).unsqueeze(2).expand(-1, embedded_x.size(1), embedded_x.size(2),-1)
        embedded_visits_number = embedded_visits_number.unsqueeze(2).expand(-1, embedded_x.size(1), embedded_x.size(2),-1)"""

        embedded_input = embedded_x.reshape(embedded_x.size(0), -1, self.projected_dim)

        #Compute the attn_mask
        batch_size = x.size(0)
        new_masks = get_encoder_mask(batch_size, 20)
        new_masks = new_masks.reshape(batch_size*self.num_heads,20, 20)

        #Apply Encoder
        encoder_output = self.transformer_encoder(embedded_input, mask = new_masks)

        # Apply transformer decoder
        #print("embedded_input.shape - ",embedded_input.shape)
        #print("encoder_output.shape - ",encoder_output.shape)
        decoder_output = self.transformer_decoder(embedded_input, encoder_output, tgt_mask=new_masks)

        # Calculate logits
        logits = self.linear(decoder_output)

        return logits

# Instantiate the model_m1
model_m1 = TranformEHR_M1(num_gender_classes, num_race_classes, num_visits=max_num_visits, num_code=len(icd_codes_types),nhead=1, num_encoder_layers=1, num_decoder_layers=1)

class TranformEHR_M2(nn.Module):
    def __init__(self, num_gender_classes, num_race_classes, num_visits, num_code, nhead, num_encoder_layers,
                 num_decoder_layers, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.concatenated_dim = embedding_dim * 6
        self.projected_dim = embedding_dim
        self.num_heads = nhead

        # Define the embedding layers
        self.visit_number_embedding = nn.Embedding(num_embeddings=num_visits, embedding_dim=embedding_dim)
        self.gender_embedding = nn.Embedding(num_gender_classes, embedding_dim)
        self.race_embedding = nn.Embedding(num_race_classes, embedding_dim)

        # Define the embeddings for other continuous features (age, position_encodings)
        self.age_embedding = nn.Linear(1, embedding_dim)  # Assuming age is a continuous feature
        self.position_encodings_embedding = nn.Linear(2, embedding_dim)  # Assuming position_encodings has 2 dimensions
        self.visit_embedding = nn.Embedding(num_embeddings=num_code, embedding_dim=embedding_dim)

        self.embedding_projection = nn.Linear(self.concatenated_dim, self.projected_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True, norm_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.projected_dim, nhead=nhead, batch_first=True, norm_first=True)

        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Linear layer to project decoder output to ICD code probabilities
        self.linear = nn.Linear(self.projected_dim, num_code)

    def forward(self, x, attn_masks, masks, visit_numbers, gender, race, age, position_encodings):

        max_visit_number = self.visit_number_embedding.num_embeddings - 1  # Get the max allowed index
        visit_numbers = torch.clamp(visit_numbers, 0, max_visit_number)  # Clamp values to valid range
        embedded_visits_number = self.visit_number_embedding(visit_numbers)
        embedded_gender = self.gender_embedding(gender)
        embedded_race = self.race_embedding(race)
        embedded_age = self.age_embedding(age.float().unsqueeze(-1))
        embedded_positional_encodings = self.position_encodings_embedding(position_encodings.float())
        embedded_x = self.visit_embedding(x)

        # Concatenate all embeddings
        embedded_positional_encodings = embedded_positional_encodings.unsqueeze(2).expand(-1, -1, embedded_x.size(2), -1)
        embedded_age = embedded_age.unsqueeze(1).unsqueeze(2).expand(-1, embedded_x.size(1), embedded_x.size(2), -1)
        embedded_race = embedded_race.unsqueeze(1).unsqueeze(2).expand(-1, embedded_x.size(1), embedded_x.size(2), -1)
        embedded_gender = embedded_gender.unsqueeze(1).unsqueeze(2).expand(-1, embedded_x.size(1), embedded_x.size(2),-1)
        embedded_visits_number = embedded_visits_number.unsqueeze(2).expand(-1, embedded_x.size(1), embedded_x.size(2),-1)

        embedded_input = torch.cat((embedded_x, embedded_visits_number, embedded_positional_encodings, embedded_age,embedded_race, embedded_gender), dim=-1)

        embedded_input = embedded_x.reshape(embedded_input.size(0), -1, self.projected_dim)

        #Calculate attn_mask
        batch_size = x.size(0)
        new_masks = get_encoder_mask(batch_size, 20)
        new_masks = new_masks.reshape(batch_size*self.num_heads,20, 20)

        #Apply Transformer Encoder
        encoder_output = self.transformer_encoder(embedded_input, mask = new_masks)

        # Apply Transformer Decoder
        #print("embedded_input.shape - ",embedded_input.shape)
        #print("encoder_output.shape - ",encoder_output.shape)
        decoder_output = self.transformer_decoder(embedded_input, encoder_output, tgt_mask=new_masks)

        logits = self.linear(decoder_output)

        return logits

# Instantiate the model_m2
model_m2 = TranformEHR_M2(num_gender_classes, num_race_classes, num_visits=max_num_visits, num_code=len(icd_codes_types),nhead=1, num_encoder_layers=1, num_decoder_layers=1)

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_recall_fscore_support

#for mac - use mps
#for all other - use cuda

#device = torch.device("cuda" if torch.backends.mps.is_available() else "cpu")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_m1.to(device)

# Define your loss function (e.g., cross-entropy)
criterion = nn.CrossEntropyLoss()

# Define your optimizer (e.g., Adam)
optimizer = torch.optim.Adam(model_m1.parameters(), lr=0.0001)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
#optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

log_interval = 5

train_losses = []  # List of training losses over epochs
eval_losses = []  # List of evaluation losses over epochs
accuracies = []  # List of accuracies over epochs
precisions = []  # List of precisions over epochs
recalls = []  # List of recalls over epochs
f1_scores = []  # List of F1-scores over epochs
topk_accuracies = []  # List of top-k accuracies over epochs
precisions_macro = []  # List of macro precisions over epochs
precisions_weighted = []  # List of weighted precisions over epochs
f1_scores_macro = []  # List of macro F1-scores over epochs
f1_scores_weighted = []  # List of weighted F1-scores over epochs
roc_auc_scores_R = []  # List of ROC AUC scores over epochs
roc_auc_scores_O = []
average_precision_scores = []  # List of average precision scores over epochs
all_probabilities_scores = []


def train_eval(model, train_data_loader, val_data_loader, epochs):

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        model.train() # Set model to training mode
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            x, attn_masks, masks, visit_numbers, gender, race, age, position_encodings = [item.to(device) for item in batch]

            # Forward pass
            logits = model(x, attn_masks, masks, visit_numbers, gender, race, age, position_encodings)

            loss = criterion(logits.view(-1, logits.size(-1)), x.view(-1))

            # Backward pass and update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # Calculate average epoch loss
        epoch_loss /= len(train_data_loader)
        print("training_epoc_loss - ", epoch_loss)
        train_losses.append(epoch_loss)

        print("starting evaluation")
        model.eval()
        with torch.no_grad():  # Disable gradient calculation for efficiency
            eval_epoch_loss = 0.0
            all_predictions = []
            all_probabilities = []
            all_targets = []
            for x, attn_masks, masks, visit_numbers, gender, race, age, position_encodings in val_data_loader:
                #x_batch_size = x.size(0)
                #if not x_batch_size < 32:
                x, attn_masks, masks, visit_numbers, gender, race, age, position_encodings = x.to(device), attn_masks.to(device), masks.to(device), \
                visit_numbers.to(device), gender.to(device), race.to(device), age.to(device), \
                position_encodings.to(device)

                # Forward pass
                print(x.shape)
                logits = model(x, attn_masks, masks, visit_numbers, gender, race, age, position_encodings)
                loss = criterion(logits.view(-1, logits.size(-1)), x.view(-1))
                eval_epoch_loss += loss.item()

                predicted_codes = torch.argmax(logits, dim=-1)
                all_predictions.append(predicted_codes.cpu().numpy().flatten())
                all_targets.append(x.cpu().numpy().flatten())

                # Convert logits to probabilities
                """predicted_probs = torch.softmax(logits, dim=-1)
                all_probabilities_scores.append(predicted_probs)
                print("predicted_probs shape", predicted_probs.shape)
                all_probabilities.append(predicted_probs.cpu().numpy())"""

            # Calculate average epoch loss
            eval_epoch_loss /= len(val_data_loader)
            eval_losses.append(eval_epoch_loss)

            # Calculate accuracy
            all_predictions = np.concatenate(all_predictions)
            """all_probabilities_np = np.concatenate(all_probabilities, axis=0)
            # Average the predicted probabilities across the sequence length dimension
            average_probabilities = np.mean(all_probabilities_np, axis=1)
            # Flatten the probabilities
            flattened_probabilities = average_probabilities.reshape(-1)"""
            all_targets = np.concatenate(all_targets)
            accuracy = accuracy_score(all_targets, all_predictions)
            accuracies.append(accuracy)

            # Calculate precision, recall, and F1-score
            precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='macro')
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            # Compute evaluation metrics
            precision_macro = precision_score(all_targets, all_predictions.round(), average='macro')
            precision_weighted = precision_score(all_targets, all_predictions.round(), average='weighted')
            precisions_macro.append(precision_macro)  # List of macro precisions over epochs
            precisions_weighted.append(precision_weighted)  # List of weighted precisions over epochs

            f1_score_macro = f1_score(all_targets, all_predictions.round(), average='macro')
            f1_score_weighted = f1_score(all_targets, all_predictions.round(), average='weighted')
            f1_scores_macro.append(f1_score_macro)  # List of macro F1-scores over epochs
            f1_scores_weighted.append(f1_score_weighted)  # List of weighted F1-scores over epochs

            #roc_auc_R = roc_auc_score(all_targets, flattened_probabilities, multi_class='ovr')
            #roc_auc_O = roc_auc_score(all_targets, flattened_probabilities, multi_class='ovo')
            #roc_auc_scores_R.append(roc_auc_R)  # List of ROC AUC scores over epochs
            #roc_auc_scores_O.append(roc_auc_O)  # List of ROC AUC scores over epochs

            #avg_precision = average_precision_score(all_targets, all_predictions) #default is - average='macro
            #average_precision_scores.append(avg_precision)  # List of average precision scores over epochs
        print(f"Epoch [{epoch + 1}/{epochs}], Average Training Loss: {epoch_loss:.4f},"
              f"Average Evaluation Loss : {eval_epoch_loss:.4f}, Accuracy: {accuracy:.4f},"
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, "
              f"Precision Macro: {precision_macro:.4f}, Precision Weighted: {precision_weighted:.4f}, "
              f"F1-score Macro: {f1_score_macro:.4f}, F1-score Weighted: {f1_score_weighted:.4f}")

#Train model_m1
import time
start = time.time()
print(start)
train_eval(model_m1, train_loader, val_loader, epochs=10)
end = time.time()
print(end)
print(end - start)

"""**Plot evaluation metrics for Model M1**"""

#Plot
import matplotlib.pyplot as plt

def plot_metrics(headline, train_losses, eval_losses, accuracies, precisions,
                 recalls, f1_scores, precisions_macro, precisions_weighted,
                 f1_scores_macro, f1_scores_weighted, roc_auc_scores_O,
                 roc_auc_scores_R, average_precision_scores):
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    plt.suptitle(headline, fontsize=16)
    # Plot losses
    plt.figure(figsize=(10, 3))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, eval_losses, label='Evaluation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Evaluation Loss')
    plt.legend()

    # Plot evaluation metrics
    plt.subplot(1, 3, 3)
    plt.plot(epochs, accuracies, label='Accuracy')
    plt.plot(epochs, precisions, label='Precision')
    plt.plot(epochs, recalls, label='Recall')
    plt.plot(epochs, f1_scores, label='F1-score')
    #plt.plot(epochs, topk_accuracies, label='Top-k Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Evaluation Metrics')
    plt.legend()

    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, precisions_macro, label='Precisions Macro')
    plt.xlabel('Epochs')
    plt.ylabel('Precisions')
    plt.title('Precisions [ Macro]')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, precisions_weighted, label='Precisions Weighted')
    plt.xlabel('Epochs')
    plt.ylabel('Precisions')
    plt.title('Precisions [ Weighted]')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, f1_scores_macro, label='F1 Macro')
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')
    plt.title('F1-Score [ Macro]')
    plt.legend()

    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, f1_scores_weighted, label='F1 Weighted')
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')
    plt.title('F1-Score [ Weighted]')
    plt.legend()

    """#average_precision_scores
    plt3.subplot(1, 3, 2)
    plt3.plot(epochs, average_precision_scores, label='Average Precision')
    plt3.xlabel('Epochs')
    plt3.ylabel('Average Precision')
    plt3.title('Average Precision')
    plt3.legend()

    plt3.subplot(1, 3, 3)
    plt3.plot(epochs, roc_auc_scores_O, label='ROC-AUC-OVO')
    plt3.xlabel('Epochs')
    plt3.ylabel('ROC-AUC-OVO')
    plt3.title('ROC-AUC [One-vs-one]')
    plt3.legend()

    plt.subplot(3, 3, 2)
    plt.plot(epochs, roc_auc_scores_R, label='ROC-AUC-OVR')
    plt.xlabel('Epochs')
    plt.ylabel('ROC-AUC-OVR')
    plt.title('ROC-AUC [One-vs-rest]')
    plt.legend()"""

    #plt.tight_layout()
    plt.show()

headlind_1 = "TransformEHR - Feature -[ICD Codes only], Num Head - 1, Encoder Layer - 1, Decoder Layer - 1, Learning Rate - 0.0001, Batch Size - 32"
plot_metrics(headlind_1, train_losses, eval_losses, accuracies, precisions, recalls,
             f1_scores, precisions_macro, precisions_weighted, f1_scores_macro,
             f1_scores_weighted, roc_auc_scores_O, roc_auc_scores_R, average_precision_scores)

"""**Create Check-Point for Model M1 for Finetuning**"""

#Create a checkpoint dictionary:
checkpoint = {
    'model_state_dict': model_m1.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': 10,  # Current training epoch
    'loss': train_losses   # Current training loss (optional)
}

filename = 'checkpoint_model1.pth'  # Create a unique filename
torch.save(checkpoint, filename)

"""**Training & Evaluation - Model M2**"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_recall_fscore_support

#for mac - use mps
#for all other - use cuda

#device = torch.device("cuda" if torch.backends.mps.is_available() else "cpu")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_m2.to(device)

# Define your loss function (e.g., cross-entropy)
criterion = nn.CrossEntropyLoss()

# Define your optimizer (e.g., Adam)
optimizer = torch.optim.Adam(model_m2.parameters(), lr=0.0001)
#optimizer = torch.optim.Adam(model_m2.parameters(), lr=1e-3, weight_decay=1e-5)
#optimizer = torch.optim.Adam(model_m2.parameters(),lr=1e-4)

log_interval = 5

train_losses = []  # List of training losses over epochs
eval_losses = []  # List of evaluation losses over epochs
accuracies = []  # List of accuracies over epochs
precisions = []  # List of precisions over epochs
recalls = []  # List of recalls over epochs
f1_scores = []  # List of F1-scores over epochs
topk_accuracies = []  # List of top-k accuracies over epochs
precisions_macro = []  # List of macro precisions over epochs
precisions_weighted = []  # List of weighted precisions over epochs
f1_scores_macro = []  # List of macro F1-scores over epochs
f1_scores_weighted = []  # List of weighted F1-scores over epochs
roc_auc_scores_R = []  # List of ROC AUC scores over epochs
roc_auc_scores_O = []
average_precision_scores = []  # List of average precision scores over epochs
all_probabilities_scores = []


def train_eval(model, train_data_loader, val_data_loader, epochs):

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        model.train() # Set model to training mode
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            x, attn_masks, masks, visit_numbers, gender, race, age, position_encodings = [item.to(device) for item in batch]

            # Forward pass
            logits = model(x, attn_masks, masks, visit_numbers, gender, race, age, position_encodings)

            loss = criterion(logits.view(-1, logits.size(-1)), x.view(-1))

            # Backward pass and update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # Calculate average epoch loss
        epoch_loss /= len(train_data_loader)
        print("training_epoc_loss - ", epoch_loss)
        train_losses.append(epoch_loss)

        print("starting evaluation")
        model.eval()
        with torch.no_grad():  # Disable gradient calculation for efficiency
            eval_epoch_loss = 0.0
            all_predictions = []
            all_probabilities = []
            all_targets = []
            for x, attn_masks, masks, visit_numbers, gender, race, age, position_encodings in val_data_loader:
                #x_batch_size = x.size(0)
                #if not x_batch_size < 32:
                x, attn_masks, masks, visit_numbers, gender, race, age, position_encodings = x.to(device), attn_masks.to(device), masks.to(device), \
                visit_numbers.to(device), gender.to(device), race.to(device), age.to(device), \
                position_encodings.to(device)

                # Forward pass
                #print(x.shape)
                logits = model(x, attn_masks, masks, visit_numbers, gender, race, age, position_encodings)
                loss = criterion(logits.view(-1, logits.size(-1)), x.view(-1))
                eval_epoch_loss += loss.item()

                predicted_codes = torch.argmax(logits, dim=-1)
                all_predictions.append(predicted_codes.cpu().numpy().flatten())
                all_targets.append(x.cpu().numpy().flatten())

                # Convert logits to probabilities
                """predicted_probs = torch.softmax(logits, dim=-1)
                all_probabilities_scores.append(predicted_probs)
                print("predicted_probs shape", predicted_probs.shape)
                all_probabilities.append(predicted_probs.cpu().numpy())"""

            # Calculate average epoch loss
            eval_epoch_loss /= len(val_data_loader)
            eval_losses.append(eval_epoch_loss)

            # Calculate accuracy
            all_predictions = np.concatenate(all_predictions)
            """all_probabilities_np = np.concatenate(all_probabilities, axis=0)
            # Average the predicted probabilities across the sequence length dimension
            average_probabilities = np.mean(all_probabilities_np, axis=1)
            # Flatten the probabilities
            flattened_probabilities = average_probabilities.reshape(-1)"""
            all_targets = np.concatenate(all_targets)
            accuracy = accuracy_score(all_targets, all_predictions)
            accuracies.append(accuracy)

            # Calculate precision, recall, and F1-score
            precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='macro')
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            # Compute evaluation metrics
            precision_macro = precision_score(all_targets, all_predictions.round(), average='macro')
            precision_weighted = precision_score(all_targets, all_predictions.round(), average='weighted')
            precisions_macro.append(precision_macro)  # List of macro precisions over epochs
            precisions_weighted.append(precision_weighted)  # List of weighted precisions over epochs

            f1_score_macro = f1_score(all_targets, all_predictions.round(), average='macro')
            f1_score_weighted = f1_score(all_targets, all_predictions.round(), average='weighted')
            f1_scores_macro.append(f1_score_macro)  # List of macro F1-scores over epochs
            f1_scores_weighted.append(f1_score_weighted)  # List of weighted F1-scores over epochs

            #roc_auc_R = roc_auc_score(all_targets, flattened_probabilities, multi_class='ovr')
            #roc_auc_O = roc_auc_score(all_targets, flattened_probabilities, multi_class='ovo')
            #roc_auc_scores_R.append(roc_auc_R)  # List of ROC AUC scores over epochs
            #roc_auc_scores_O.append(roc_auc_O)  # List of ROC AUC scores over epochs

            #avg_precision = average_precision_score(all_targets, all_predictions) #default is - average='macro
            #average_precision_scores.append(avg_precision)  # List of average precision scores over epochs
        print(f"Epoch [{epoch + 1}/{epochs}], Average Training Loss: {epoch_loss:.4f},"
              f"Average Evaluation Loss : {eval_epoch_loss:.4f}, Accuracy: {accuracy:.4f},"
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, "
              f"Precision Macro: {precision_macro:.4f}, Precision Weighted: {precision_weighted:.4f}, "
              f"F1-score Macro: {f1_score_macro:.4f}, F1-score Weighted: {f1_score_weighted:.4f}")

#Train model_m2
import time
start = time.time()
print(start)
train_eval(model_m2, train_loader, val_loader, epochs=10)
end = time.time()
print(end)
print(end - start)

"""**Plot evaluation metrics for Model M2**"""

# Model #2
headlind_2 = "TransformEHR - Feature - all features], Num Head - 1, Encoder Layer - 1, Decoder Layer - 1, Learning Rate - 0.0001, Batch Size - 32"
plot_metrics(headlind_2, train_losses, eval_losses, accuracies, precisions, recalls,
             f1_scores, precisions_macro, precisions_weighted, f1_scores_macro,
             f1_scores_weighted, roc_auc_scores_O, roc_auc_scores_R, average_precision_scores)

"""**Create CheckPoints for Model M2 for Finetuning**"""

#Create a checkpoint dictionary:
checkpoint = {
    'model_state_dict': model_m2.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': 10,  # Current training epoch
    'loss': train_losses   # Current training loss (optional)
}

filename = 'checkpoint_model2.pth'  # Create a unique filename
torch.save(checkpoint, filename)

"""**Disease/outcome agnostic prediction: AUROC scores on different pretraining objectives for the 10 common and 10 uncommon diseases**"""

common_outcomes["ICD-10-CM Code ID"] = [icd_codes_types[code] for code in common_outcomes["ICD-10-CM Code"]]
print(common_outcomes)

uncommon_outcomes["ICD-10-CM Code ID"] = [icd_codes_types[code] for code in uncommon_outcomes["ICD-10-CM Code"]]
print(uncommon_outcomes)

import torch
from sklearn.metrics import precision_score, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt


def evaluate_model(model, data_loader, outcomes):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  model.eval()
  disease_codes = outcomes['ICD-10-CM Code ID']
  num_diseases = len(disease_codes)
  aurocs = []

  predictions = {code: [] for code in disease_codes}
  targets = {code: [] for code in disease_codes}

  num = 0
  with torch.no_grad():
    for x, attn_masks, masks, visit_numbers, gender, race, age, position_encodings in data_loader:
      x = x.to(device)
      # Run model
      logits = model(x, attn_masks, masks, visit_numbers, gender, race, age, position_encodings)
      probs = torch.sigmoid(logits[:, -1, :])  # Get the last visit probabilities

      # Extract ICD codes for the last visit for each sample in the batch
      last_visit_codes = x[:, -1, :]  # Shape: (batch_size, num_icd_codes_per_visit)

      for i, code_id in enumerate(disease_codes):
        # Check if each sample's last visit ICD codes contain the current disease code
        code_mask = (last_visit_codes == code_id).any(dim=1)  # Shape: (batch_size)
        batch_targets = code_mask.float().cpu().numpy()
        # if (num == 0):
        #   print("code_id:", code_id)
        #   print("last_visit_codes:", last_visit_codes)
        #   print("last_visit_codes shape:", last_visit_codes.shape)
        #   print("probs:", probs)
        #   print("probs shape:", probs.shape)
        #   print("batch_targets:", batch_targets, batch_targets.shape)

        # Collect targets and predictions
        targets[code_id].extend(batch_targets)
        predictions[code_id].extend(probs[:, i].cpu().numpy())
      num += 1

  # Calculate AUROC for each disease
  for code_id in disease_codes:
    if len(np.unique(targets[code_id])) > 1:
      auroc = roc_auc_score(targets[code_id], predictions[code_id])
      aurocs.append((outcomes['Description'][disease_codes.index(code_id)], auroc))

  return aurocs

"""**Model M1 - AUROC scores on different pretraining objectives for the 10 common and 10 uncommon diseases**


"""

data_loader = load_data(dataset, batch_size = 32, shuffle=False)

#Load Model M1

model_m1_checkpoint_path = 'checkpoint_model1.pth'

# Load model checkpoint
model_m1_checkpoint = torch.load(model_m1_checkpoint_path)
model_m1_state_dict = model_m1_checkpoint['model_state_dict']

model_m1 = TranformEHR_M1(num_gender_classes, num_race_classes, num_visits=max_num_visits, num_code=len(icd_codes_types),nhead=1, num_encoder_layers=1, num_decoder_layers=1)

# Load the model's state dictionary
model_m1.load_state_dict(model_m1_state_dict)

"""**Compute AUROC for 10 common diseases outcomes**


"""

model_m1_aurocs_common = evaluate_model(model_m1, data_loader, common_outcomes)

import pandas as pd
#from IPython.display import display

df_results = pd.DataFrame(model_m1_aurocs_common, columns=['Disease', 'AUROC Score'])
df_results['AUROC Score'] = df_results['AUROC Score'].round(2)
print(df_results)
#display(df_results)

"""**Compute AUROC for 10 uncommon diseases outcomes**"""

model_m1_aurocs_uncommon = evaluate_model(model_m1, data_loader, uncommon_outcomes)

import pandas as pd
#from IPython.display import display

df_results = pd.DataFrame(model_m1_aurocs_uncommon, columns=['Disease', 'AUROC Score'])
df_results['AUROC Score'] = df_results['AUROC Score'].round(2)
print(df_results)
#display(df_results)

"""**Model M2 - AUROC scores on different pretraining objectives for the 10 common and 10 uncommon diseases**"""

#Load Model M2

model_m2_checkpoint_path = 'checkpoint_model2.pth'

# Load model checkpoint
model_m2_checkpoint = torch.load(model_m2_checkpoint_path)
model_m2_state_dict = model_m2_checkpoint['model_state_dict']

model_m2 = TranformEHR_M2(num_gender_classes, num_race_classes, num_visits=max_num_visits, num_code=len(icd_codes_types),nhead=1, num_encoder_layers=1, num_decoder_layers=1)

# Load the model's state dictionary
model_m2.load_state_dict(model_m2_state_dict)

"""**Compute AUROC for 10 common diseases outcomes**"""

model_m2_aurocs_common = evaluate_model(model_m2, data_loader, common_outcomes)

import pandas as pd
#from IPython.display import display

df_results = pd.DataFrame(model_m2_aurocs_common, columns=['Disease', 'AUROC Score'])
df_results['AUROC Score'] = df_results['AUROC Score'].round(2)
print(df_results)
#display(df_results)

model_m2_aurocs_uncommon = evaluate_model(model_m2, data_loader, uncommon_outcomes)

import pandas as pd
#from IPython.display import display

df_results = pd.DataFrame(model_m2_aurocs_uncommon, columns=['Disease', 'AUROC Score'])
df_results['AUROC Score'] = df_results['AUROC Score'].round(2)
print(df_results)
#display(df_results)

"""from tabulate import tabulate

# Define the data
data = [
    ["Average Training Loss", 0.0790, 0.0781],
    ["Average Evaluation Loss", 0.4849, 0.4694],
    ["Accuracy", 0.9720, 0.9726],
    ["Precision", 0.7449, 0.7487],
    ["Recall", 0.7678, 0.7710],
    ["F1-score", 0.7527, 0.7562],
    ["Precision Macro", 0.7449, 0.7487],
    ["Precision Weighted", 0.9649, 0.9661],
    ["F1-score Macro", 0.7527, 0.7562],
    ["F1-score Weighted", 0.9677, 0.9685]
]

# Print the table
print(tabulate(data, headers=["Metric", "Model-1 [Only ICD Codes] ", "Model-2 [ICD Codes + Demographic Info + Visit Dates]"]))"""