import requests

sentence = {
    'Hello! How are you doing?'
}

url = f'http://127.0.0.1:9696/translate'
response = requests.post(url=url, json=sentence)
print(response.json())