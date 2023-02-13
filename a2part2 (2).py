import os
import re
import sys
import string
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(1)

_labels = ['eng', 'fra', 'deu', 'ita', 'spa']
label_to_int = {label:index for (index, label) in enumerate(_labels)}
int_to_label = {index:label for (index, label) in enumerate(_labels)}


class LangDataset(Dataset):
    """
    Define a pytorch dataset class that accepts a text path, and optionally label path and
    a vocabulary (depends on your implementation). This class holds all the data and implement
    a __getitem__ method to be used by a Python generator object or other classes that need it.

    DO NOT shuffle the dataset here, and DO NOT pad the tensor here.
    """
    def __init__(self, text_path, label_path=None, vocab=None):
        """
        Read the content of vocab and text_file
        Args:
            vocab (string): Path to the vocabulary file.
            text_file (string): Path to the text file.
        """
        train_file_open = open(text_path, 'r')
        train_file_data = train_file_open.readlines()
        
        if label_path:
            trainLabel_file_open = open(label_path, 'r')
            trainLabel_file_data = trainLabel_file_open.readlines()
            train_label_list = trainLabel_file_data
        else:
            train_label_list = []

        if(vocab):
            self.text_vocab = vocab
        else:
            self.text_vocab = self.create_text_vocab(train_file_data)

        def transform_line(line):
            list_encode = []
            char_line = line.strip().replace(" ", "")
            for index in range(len(char_line) - 1):
                bigram = char_line[index] + char_line[index+1]
                if bigram in self.text_vocab:
                    code = self.text_vocab[bigram]
                    list_encode.append(code)
            return list_encode


        self.texts = [transform_line(line) for line in train_file_data]
        self.labels = [label_to_int[label.strip()] for label in train_label_list] 

        #self.texts = [self.make_bow_vector(x) for x in self.texts]
        #self.labels = [self.make_target(y) for y in self.labels]

    def create_text_vocab(self, data):
        bigram_dict = {}
        c = 0
 
        for line in data:
            if type(line) != str:
                continue
            despaced_line = line.replace(" ", "")
            
            for i in range(len(despaced_line)-1):
                curr = despaced_line[i]
                next = despaced_line[i+ 1]
                s = curr + next
                
                if s not in bigram_dict:
                    bigram_dict[s] = c
                    c+=1

        text_vocab = bigram_dict
        # print(text_vocab, len(text_vocab))
        return text_vocab

    def create_label_vocab(self, data):
        label_vocab = {}
        for label in data:
            if label not in label_vocab:
                label_vocab[label] = len(label_vocab)
        
        return label_vocab

    def make_bow_vector(self, data):
        vec = torch.zeros(len(self.text_vocab))
        for word in data:
            vec[self.text_vocab[word]] += 1
        return vec.view(1, -1)
    
    def make_target(self, label):
        return torch.LongTensor([self.label_vocab[label]])

    def vocab_size(self):
        """
        A function to inform the vocab size. The function returns two numbers:
            num_vocab: size of the vocabulary
            num_class: number of class labels
        """
        # Stores {label_i : index} - label_vocab
        label_to_ix = self.create_label_vocab(self.labels)

        # Stores {word1: index1, word2: index2 ...} - text_vocab
        word_to_ix = self.create_text_vocab(self.texts)
                
        num_vocab = len(self.text_vocab)
        num_class = len(_labels)

        return num_vocab, num_class
    
    def __len__(self):
        """
        Return the number of instances in the data
        """
        return len(self.texts)

    def __getitem__(self, i):
        """
        Return the i-th instance in the format of:
            (text, label)
        Text and label should be encoded according to the vocab (word_id).

        DO NOT pad the tensor here, do it at the collator function.
        """
        text = self.texts[i]
        label = self.labels[i] if self.labels else torch.empty(1)

        return text, label


class Model(nn.Module):
    """
    Define a model that with one embedding layer with dimension 16 and
    a feed-forward layers that reduce the dimension from 16 to 200 with ReLU activation
    a dropout layer, and a feed-forward layers that reduce the dimension from 200 to num_class
    """
    def __init__(self, num_vocab, num_class, dropout=0.3):
        super().__init__()
        # define your model here
        embedding_dim = 16
        self.embeddings = torch.nn.Embedding(num_vocab, embedding_dim)
        self.embeddings.weight.data.uniform_(-1,1)
        self.linear1 = nn.Linear(embedding_dim, 200)
        self.linear2 = nn.Linear(200, num_class)
        self.dropoutLayer = nn.Dropout(dropout)

    def forward(self, x):
        embeds = self.embeddings(x)#.view((1, -1))
        avg_embed = embeds.sum(dim = 1) / (embeds != 0).sum(dim = 1)
        out = F.relu(self.linear1(avg_embed))
        dropout = self.dropoutLayer(out)
        out_2 = self.linear2(dropout)

        # print(out.dim)
        # define the forward function here
        return F.softmax(out_2)


def collator(batch):
    """
    Define a function that receives a list of (text, label) pair
    and return a pair of tensors:
        texts: a tensor that combines all the text in the mini-batch, pad with 0
        labels: a tensor that combines all the labels in the mini-batch
    """
    texts_unpadded = [text for (text, label) in batch]
    labels = [label for (text, label) in batch]

    # Find length of longest tensor in list
    max_len = len(max(texts_unpadded, key = len))
    
    def pad(text_tensor):
        curr_tensor = torch.tensor(text_tensor, dtype = torch.long)
        zeroes = torch.zeros(max_len - curr_tensor.shape[0], dtype = torch.long)
        new = torch.cat((curr_tensor, zeroes))
        return new
    # Stack all the new padded tensors to form the result
    texts_padded =  torch.stack([pad(tensor) for tensor in texts_unpadded])  
    texts = texts_padded
    #print(len(labels))
    labels = torch.tensor(labels)#, dtype = torch.long)

    return texts, labels

    
def train(model, dataset, batch_size, learning_rate, num_epoch, device='cpu', model_path=None):
    """
    Complete the training procedure below by specifying the loss function
    and optimizers with the specified learning rate and specified number of epoch.
    
    Do not calculate the loss from padding.
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)

    # assign these variables
    criterion = None
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_function = F.cross_entropy

    start = datetime.datetime.now()
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        for step, data in enumerate(data_loader, 0):
            # get the inputs; data is a tuple of (inputs, labels)
            texts = data[0].to(device)
            labels = data[1].to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            #vec = make_bow_vector(step, ).to(device)
            #target = make_target(label, label_to_ix).to(device)
            # do forward propagation
            # print(texts)
            #print(model.embeddings(texts))
        
            log_probs = model.forward(texts)
            # print(log_probs)
            # quit()
            # do loss calculation
            loss = loss_function(log_probs, labels)
            running_loss += loss.item()

            # do backward propagation
            loss.backward()
            # do parameter optimization step
            optimizer.step()
            # calculate running loss value for non padding

            # print loss value every 100 steps and reset the running loss
            if step % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, step + 1, running_loss / 100))
                running_loss = 0.0

    end = datetime.datetime.now()
    
    # define the checkpoint and save it to the model path
    # tip: the checkpoint can contain more than just the model
    checkpoint = {
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': loss,
              'vocab': {'dict': dataset.text_vocab, 'size': dataset.vocab_size()[0]}
              }
    torch.save(checkpoint, model_path)

    print('Model saved in ', model_path)
    print('Training finished in {} minutes.'.format((end - start).seconds / 60.0))


def test(model, dataset, class_map, device='cpu'):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=20, collate_fn=collator, shuffle=False)
    labels = []
    with torch.no_grad():
        for data in data_loader:
            texts = data[0].to(device)
            outputs = model(texts).cpu()
            # get the label predictions
            labels.extend([int_to_label[i.item()] for i in torch.argmax(outputs, dim=1)])

    return labels


def main(args):
    if torch.cuda.is_available():
        device_str = 'cuda:{}'.format(0)
    else:
        device_str = 'cpu'
    device = torch.device(device_str)
    
    assert args.train or args.test, "Please specify --train or --test"
    if args.train:
        assert args.label_path is not None, "Please provide the labels for training using --label_path argument"
        dataset = LangDataset(args.text_path, args.label_path)
        num_vocab, num_class = dataset.vocab_size()
        model = Model(num_vocab, num_class).to(device)
        
        # you may change these hyper-parameters
        learning_rate = 0.0001
        batch_size = 2
        num_epochs = 80

        train(model, dataset, batch_size, learning_rate, num_epochs, device, args.model_path)
    if args.test:
        assert args.model_path is not None, "Please provide the model to test using --model_path argument"
       
        checkpoint = torch.load(args.model_path)
        # create the test dataset object using LangDataset class
        dataset = LangDataset(args.text_path, args.label_path, vocab = checkpoint['vocab']['dict'])
        
        # initialize and load the model
        model = Model(checkpoint['vocab']['size'], 5)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        # the lang map should contain the mapping between class id to the language id (e.g. eng, fra, etc.)
        lang_map = int_to_label


        # run the prediction
        preds = test(model, dataset, lang_map, device)
        
        # write the output
        with open(args.output_path, 'w', encoding='utf-8') as out:
            out.write('\n'.join(preds))
    print('\n==== A2 Part 2 Done ====')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_path', help='path to the text file')
    parser.add_argument('--label_path', default=None, help='path to the label file')
    parser.add_argument('--train', default=False, action='store_true', help='train the model')
    parser.add_argument('--test', default=False, action='store_true', help='test the model')
    parser.add_argument('--model_path', required=True, help='path to the output file during testing')
    parser.add_argument('--output_path', default='out.txt', help='path to the output file during testing')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    main(args)