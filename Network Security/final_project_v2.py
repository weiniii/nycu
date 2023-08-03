import requests
import datetime
from urllib.parse import urlencode
import base64
import pip
import time
import logging
try:
    from spotipy.oauth2 import SpotifyClientCredentials
    import spotipy
except:
    package_name = 'spotipy==2.22.0'
    pip.main(['install', package_name])
    from spotipy.oauth2 import SpotifyClientCredentials
    import spotipy

from selenium import webdriver

logging.basicConfig(filename='311554038_V2.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(message)s')

class SpotifyAPI(object):
    access_token = None
    access_token_expires = datetime.datetime.now()
    access_token_did_expire = True
    client_id = None
    client_secret = None
    token_url = "https://accounts.spotify.com/api/token"
    
    def __init__(self, client_id, client_secret, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_id = client_id
        self.client_secret = client_secret

    def get_client_credentials(self):
        """
        Returns a base64 encoded string
        """
        client_id = self.client_id
        client_secret = self.client_secret
        if client_secret == None or client_id == None:
            raise Exception("You must set client_id and client_secret")
        client_creds = f"{client_id}:{client_secret}"
        client_creds_b64 = base64.b64encode(client_creds.encode())
        return client_creds_b64.decode()
    
    def get_token_headers(self):
        client_creds_b64 = self.get_client_credentials()
        return {
            "Authorization": f"Basic {client_creds_b64}"
        }
    
    def get_token_data(self):
        return {
            "grant_type": "client_credentials"
        } 
    
    def perform_auth(self):
        token_url = self.token_url
        token_data = self.get_token_data()
        token_headers = self.get_token_headers()
        r = requests.post(token_url, data=token_data, headers=token_headers)
        if r.status_code not in range(200, 299):
            return False
        data = r.json()
        now = datetime.datetime.now()
        access_token = data['access_token']
        expires_in = data['expires_in'] # seconds
        expires = now + datetime.timedelta(seconds=expires_in)
        self.access_token = access_token
        self.access_token_expires = expires
        self.access_token_did_expire = expires < now
        return True

print('You need to authentucate first.')
while True:
    print('Please input your client id: ')
    client_id = input()
    # logging.info('A user input a client id')

    print('Please input your client secret: ')
    client_secret = input()
    # logging.info('A user input a client secret')

    spotify = SpotifyAPI(client_id, client_secret)
    # logging.info('Authenticating')

    if spotify.perform_auth():
        print('Successful Authentication !!!')
        # logging.info('Successful login')
        break
    else:
        print('Fail to authenticate !!!')
        # logging.warning('Fail to login, and restart to login')
        print('Restart to input a correct client id and secret !!!')

spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))


while True:
    print('You want to search a playlist by "playlist" or "track"' )
    way = input()
    # logging.info('Select one way to find a playlist')
    if way not in ["playlist", "track"]:
        print('Input error ! Break !')
        # logging.warning('Select an error way, and would restart to select')
    elif way == 'playlist':
        my_private_playlist_name  = 'spotify:playlist:1UZwPMf71BHbanYUn8Y6vq'
        my_private_playlist = spotify.playlist(my_private_playlist_name)
        print('playlist name: '+str(my_private_playlist['name']))
        # logging.info(f'Select a "correct" way', {my_private_playlist_name})
        url = list(my_private_playlist['external_urls'].values())[0]
        break
    else:
        my_private_playlist_from_track_name  = 'spotify:track:../playlists/1UZwPMf71BHbanYUn8Y6vq'
        my_private_playlist_from_track = spotify.track(my_private_playlist_from_track_name)
        print('playlist name: '+str(my_private_playlist_from_track['name']))
        # logging.info('Select a "track" way')
        url = list(my_private_playlist_from_track['external_urls'].values())[0]
        # logging.warning(f'Select a "Path Traversal" way, {my_private_playlist_from_track_name}')
        break
    
print('Do you want to open the website of this playlist?(yes/no)')
want = input()
# logging.info('If opening the website')
if want not in ['yes', 'no']:
    print('Input error ! Break !')
    # logging.warning('Input error')
elif want == 'yes':
    driver = webdriver.Chrome('chromedriver')
    driver.implicitly_wait(10)
    driver.get(url)
    # logging.info('Open')
    time.sleep(600)
else:
    print('OK! Goodbye!')
    # logging.info('Not open')

# logging.info('Logout. Exit')
exit()
