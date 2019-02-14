import json
import pylyrics3 as pl
import requests as r
import base64


def getToken():
	res = r.post(url = tokenURL, headers = tokenheader, data = {'grant_type': 'client_credentials'})
	#print(res)
	data = res.json()
	return data["access_token"]

f = open('ClientAndPlaylistID.txt', 'r')

clientID = str.encode(f.readline()[:-1]) #has to be byte representation
playlistID = f.readline()
print(playlistID)
tokenheader = {'Authorization': b'Basic ' + base64.b64encode(clientID)}
tokenURL = 'https://accounts.spotify.com/api/token'
token = getToken()

num_tracks = 10000 
playlistHeader = {'Authorization' : 'Bearer ' + token}

offset = 0
got = 0
dgot = 0
for i in range(int(num_tracks / 100) + 1):

	offsetString = '?offset=' + str(offset)

	res = r.get(url = "https://api.spotify.com/v1/playlists/" + playlistID + '/tracks' + offsetString, headers = playlistHeader)
	print(res)
	data = res.json()

	f = open('D:\LyricLSTM\\lyrics.txt', "a")
	f2 = open('D:\LyricLSTM\\didntget.txt',"a")

	for t in data['items']:
		try:
			f.write(pl.get_song_lyrics(t['track']['artists'][0]['name'], t['track']['name']) + "\n")
			got += 1
		except:
			dgot += 1
			f2.write(t['track']['name'] + ' - ' + t['track']['artists'][0]['name'] + '\n')
	offset += 100
	print(got, got + dgot)