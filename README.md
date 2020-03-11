# LyricLSTM
Generate lyrics using an LSTM from spotify playlists


# HOW TO USE

replace ClientID:SecretID and PlaylistID in ClientAndPLaylistID.txt file with your own IDs

run getLyrics.py (the more lyrics the better the results, so use large playlists)
  
run word2vec.py, train.py, and then generate.py


Note: The model often gets stuck repeating words. I am not sure I am feeding the right input at each step
