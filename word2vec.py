import gzip
import gensim 
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

data_file="angsty2.lyrics"

def read_input(input_file):
    
    with open(input_file, 'rb') as f:
        for i, line in enumerate (f): 

            if (i%10000==0):
                logging.info ("read {0} lines".format (i))
            yield gensim.utils.simple_preprocess(line, min_len = 0) #to lowercase and strips most sppecial characters, outputs list

# read the tokenized posts into a list
# each review item becomes a serries of words
# so this becomes a list of lists
documents = list (read_input (data_file))
logging.info ("Done reading data file")

model = gensim.models.Word2Vec(documents, size=100, window=3, min_count=1, workers=10)
model.train(documents,total_examples=len(documents), epochs=10)

model.save("dim100.w2v")
