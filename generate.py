import torch
import torch.nn as nn
from gensim.models import Word2Vec
from random import randint

class LSTM(nn.Module):
	
	def __init__(self, embedding_dim, seq_length, hidden_dim, target_size):
	
		super(LSTM, self).__init__()
	
		self.hidden_dim = hidden_dim
		self.input_size = embedding_dim
		self.seq_length = seq_length
		self.lstm = nn.LSTM(self.input_size, hidden_dim)
		self.hidden2target = nn.Linear(hidden_dim, target_size)
		self.hidden = self.init_hidden()


	def init_hidden(self):
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		return (torch.zeros(1, self.seq_length, self.hidden_dim), torch.zeros(1, self.seq_length, self.hidden_dim))

	def forward(self, inputs):
		lstm_out, self.hidden = self.lstm(inputs, self.hidden)
		return lstm_out, self.hidden2target( lstm_out[0][-1].view(-1)) #len of inputs is going to have to be length of one of the demensions of inputs


embedding_size = 80
seq_length = 6
hidden_dim = 1200

model = LSTM(embedding_size, seq_length, hidden_dim, embedding_size)
seed_str = 'not it bad not time me'

model.load_state_dict(torch.load('model'))
model.eval()

w2v = Word2Vec.load('dim100.w2v')
inp = torch.tensor(w2v[seed_str.split()], dtype=torch.float32).view(1, seq_length,-1) 

for i in range(1000):
	lstm_out, out = model.forward(inp)
	print(lstm_out)
	print(w2v.most_similar(positive=[out.detach().numpy()])[randint(0, 3)][0], end = ' ') #picks randomly from top 4 most similar, less coherent results but prevents getting stuck
	inp = torch.cat( (inp[0][1:], lstm_out[0][0].view(1,-1)), 0).view(1,seq_length,-1)
	#print(lstm_out)
	#inp = lstm_out