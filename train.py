import torch
import torch.nn as nn
from gensim.models import Word2Vec
import numpy as np
import gensim


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
		# The axis semantics are (num_layers, minibatch_size, hidden_dim)
		return (torch.zeros(1, self.seq_length, self.hidden_dim), torch.zeros(1, self.seq_length, self.hidden_dim))

	def forward(self, inputs):
		lstm_out, self.hidden = self.lstm(inputs, self.hidden)
		return self.hidden2target( lstm_out[0][-1].view(-1)) #len of inputs is going to have to be length of one of the demensions of inputs


#no batches implemented yet
def trainGenerator(text_file, w2v, vector_dim, batch_size, seq_length):

	X = np.zeros(vector_dim * seq_length)
	for line in text_file:
		staging = gensim.utils.simple_preprocess(line, min_len = 0)
		while len(staging) > 1:
			try:
				Y = w2v[staging[-1]]
			except:
				Y = w2v["a"] #if word isnt it vocabulary just 'a'
				print("word out of vocab \n")
			X = X[vector_dim:]
			try:
				X = np.append(X, w2v[staging[-2]] )
			except:
				X = np.append(X, w2v["a"]) #if word isnt it vocabulary just 'a'
				print("word out of vocab \n")
			staging = staging[:-2]
			yield torch.tensor(X, dtype=torch.float32).view(1, seq_length,-1), torch.tensor(Y, dtype=torch.float32)
				
	
def main():

	embedding_size = 80
	seq_length = 6
	hidden_dim = 1200

	model = LSTM(embedding_size, seq_length, hidden_dim, embedding_size)
	loss_function = nn.MSELoss()
	optimizer = torch.optim.SGD(model.parameters(), lr = 0.5)
	w2v = Word2Vec.load('dim80-2.w2v')
	f = open('angsty.lyrics', 'r')

	gen = trainGenerator(f, w2v, embedding_size, 1, seq_length)


	sum_loss = 0
	for epoch in range(20000):
		
		model.zero_grad()
		model.hidden = model.init_hidden() #ahhhh I dont know

		x, y = next(gen)
		pred = model.forward(x)
		loss = loss_function(pred, y)
		loss.backward()
		optimizer.step()
		sum_loss += loss.item()

		#batch
		if (epoch % 40) == 0:
				print("loss = ", sum_loss / 40)
				sum_loss = 0
		if (epoch % 5000) == 0:
			torch.save(model.state_dict(), 'model3-Tokyo-Drift')





if __name__ == "__main__":
    main()