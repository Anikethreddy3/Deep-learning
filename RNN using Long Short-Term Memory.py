

# Commented out IPython magic to ensure Python compatibility.
# %load_ext watermark
# %watermark -a 'Sebastian Raschka' -v -p torch,torchtext
# %env CUBLAS_WORKSPACE_CONFIG=:4096:8
import torch
import torch.nn.functional as F
import torchtext
import time
import random
import pandas as pd
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.deterministic = True

"""## General Settings"""

RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)

VOCABULARY_SIZE = 20000
LEARNING_RATE = 0.005
BATCH_SIZE = 128
NUM_EPOCHS = 15
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_CLASSES = 4

"""## Download Dataset

The following cells will download the IMDB movie review dataset (http://ai.stanford.edu/~amaas/data/sentiment/) for positive-negative sentiment classification in as CSV-formatted file:

Check that the dataset looks okay:
"""

!gunzip uci-news-aggregator.csv.gz

df = pd.read_csv('uci-news-aggregator.csv')
df.head()

df = df[['TITLE', 'CATEGORY']]
# df.columns= ['TITLE', 'LABEL_COLUMN_NAME']
df.to_csv('uci-news-aggregator.csv', index=None)

df = pd.read_csv('uci-news-aggregator.csv')
df.head()
df.columns[1]

df.head()

del df

"""## Prepare Dataset with Torchtext"""

!pip install spacy

! pip install torchtext==0.9

"""Download English vocabulary via:
    
- `python -m spacy download en_core_web_sm`

Define the Label and Text field formatters:
"""

### Defining the feature processing

TEXT = torchtext.legacy.data.Field(
    tokenize='spacy', # default splits on whitespace
    tokenizer_language='en_core_web_sm',
    include_lengths=True # NEW
)

### Defining the label processing

LABEL = torchtext.legacy.data.LabelField(dtype=torch.long)

"""Process the dataset:"""

fields = [('TITLE', TEXT), ('CATEGORY', LABEL)]

dataset = torchtext.legacy.data.TabularDataset(
    path='uci-news-aggregator.csv', format='csv',
    skip_header=True, fields=fields)

"""## Split Dataset into Train/Validation/Test

Split the dataset into training, validation, and test partitions:
"""

train_data, test_data = dataset.split(
    split_ratio=[0.8, 0.2],
    random_state=random.seed(RANDOM_SEED))

print(f'Num Train: {len(train_data)}')
print(f'Num Test: {len(test_data)}')

train_data, valid_data = train_data.split(
    split_ratio=[0.85, 0.15],
    random_state=random.seed(RANDOM_SEED))

print(f'Num Train: {len(train_data)}')
print(f'Num Validation: {len(valid_data)}')

print(vars(train_data.examples[10]))

"""## Build Vocabulary

Build the vocabulary based on the top "VOCABULARY_SIZE" words:
"""

TEXT.build_vocab(train_data, max_size=VOCABULARY_SIZE)
LABEL.build_vocab(train_data)

print(f'Vocabulary size: {len(TEXT.vocab)}')
print(f'Number of classes: {len(LABEL.vocab)}')

"""- 25,002 not 25,000 because of the `<unk>` and `<pad>` tokens
- PyTorch RNNs can deal with arbitrary lengths due to dynamic graphs, but padding is necessary for padding sequences to the same length in a given minibatch so we can store those in an array

**Look at most common words:**
"""

print(TEXT.vocab.freqs.most_common(20))

"""**Tokens corresponding to the first 10 indices (0, 1, ..., 9):**"""

print(TEXT.vocab.itos[:10]) # itos = integer-to-string

"""**Converting a string to an integer:**"""

print(TEXT.vocab.stoi['the']) # stoi = string-to-integer

"""**Class labels:**"""

print(LABEL.vocab.stoi)

"""**Class label count:**"""

LABEL.vocab.freqs

"""## Define Data Loaders"""

train_loader, valid_loader, test_loader = \
    torchtext.legacy.data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=True, # NEW. necessary for packed_padded_sequence
             sort_key=lambda x: len(x.TITLE),
        device=DEVICE
)

"""Testing the iterators (note that the number of rows depends on the longest document in the respective batch):"""

print('Train')
for batch in train_loader:
    print(f'Text matrix size: {batch.TITLE[0].size()}')
    print(f'Target vector size: {batch.CATEGORY.size()}')
    break

print('\nValid:')
for batch in valid_loader:
    print(f'Text matrix size: {batch.TITLE[0].size()}')
    print(f'Target vector size: {batch.CATEGORY.size()}')
    break

print('\nTest:')
for batch in test_loader:
    print(f'Text matrix size: {batch.TITLE[0].size()}')
    print(f'Target vector size: {batch.CATEGORY.size()}')
    break

"""## Model"""

class RNN(torch.nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        #self.rnn = torch.nn.RNN(embedding_dim,
        #                        hidden_dim,
        #                        nonlinearity='relu')
        self.rnn = torch.nn.LSTM(embedding_dim,
                                 output_dim)

        self.fc = torch.nn.Linear(hidden_dim, output_dim)


    def forward(self, text, text_length):
        # text dim: [sentence length, batch size]

        embedded = self.embedding(text)
        # ebedded dim: [sentence length, batch size, embedding dim]

        ## NEW
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, text_length.to('cpu'))

        packed_output, (hidden, cell) = self.rnn(packed)
        # output dim: [sentence length, batch size, hidden dim]
        # hidden dim: [1, batch size, hidden dim]

        hidden.squeeze_(0)
        # hidden dim: [batch size, hidden dim]

        # output = self.fc(hidden)
        return hidden

!pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

torch.manual_seed(RANDOM_SEED)
model = RNN(input_dim=len(TEXT.vocab),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=NUM_CLASSES # could use 1 for binary classification
)

model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

"""## Training"""

def compute_accuracy(model, data_loader, device):

    with torch.no_grad():

        correct_pred, num_examples = 0, 0

        for batch_idx, batch_data in enumerate(data_loader):

            # NEW
            features, text_length = batch_data.TITLE
            targets = batch_data.CATEGORY.to(DEVICE)

            logits = model(features, text_length)
            _, predicted_labels = torch.max(logits, 1)

            num_examples += targets.size(0)

            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_idx, batch_data in enumerate(train_loader):

        # NEW
        features, text_length = batch_data.TITLE
        labels = batch_data.CATEGORY.to(DEVICE)

        ### FORWARD AND BACK PROP
        logits = model(features, text_length)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()

        loss.backward()

        ### UPDATE MODEL PARAMETERS
        optimizer.step()

        ### LOGGING
        if not batch_idx % 50:
            print (f'Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d} | '
                   f'Batch {batch_idx:03d}/{len(train_loader):03d} | '
                   f'Loss: {loss:.4f}')

    with torch.set_grad_enabled(False):
        print(f'training accuracy: '
              f'{compute_accuracy(model, train_loader, DEVICE):.2f}%'
              f'\nvalid accuracy: '
              f'{compute_accuracy(model, valid_loader, DEVICE):.2f}%')

    print(f'Time elapsed: {(time.time() - start_time)/60:.2f} min')

print(f'Total Training Time: {(time.time() - start_time)/60:.2f} min')
print(f'Test accuracy: {compute_accuracy(model, test_loader, DEVICE):.2f}%')

print(LABEL.vocab.stoi)

import spacy


nlp = spacy.blank("en")

def predict(model, sentence):

    model.eval()

    with torch.no_grad():
        tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
        indexed = [TEXT.vocab.stoi[t] for t in tokenized]
        length = [len(indexed)]
        tensor = torch.LongTensor(indexed).to(DEVICE)
        tensor = tensor.unsqueeze(1)
        length_tensor = torch.LongTensor(length)
        predict_probas = torch.nn.functional.softmax(model(tensor, length_tensor), dim=1)
        predicted_label_index = torch.argmax(predict_probas)
        predicted_label_proba = torch.max(predict_probas)
        return predicted_label_index.item(), predicted_label_proba.item()


class_mapping = LABEL.vocab.stoi
inverse_class_mapping = {v: k for k, v in class_mapping.items()}

predicted_label_index, predicted_label_proba = \
    predict(model, "This is such an awesome movie, I really love it!")
predicted_label = inverse_class_mapping[predicted_label_index]

print(f'Predicted label index: {predicted_label_index}'
      f' | Predicted label: {predicted_label}'
      f' | Probability: {predicted_label_proba} ')

predicted_label_index, predicted_label_proba = \
    predict(model, "I really hate this movie. It is really bad and sucks!")
predicted_label = inverse_class_mapping[predicted_label_index]

print(f'Predicted label index: {predicted_label_index}'
      f' | Predicted label: {predicted_label}'
      f' | Probability: {predicted_label_proba} ')