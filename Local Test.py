import requests
import json

# enter URL
url_upload = "http:.../upload"
url_train = "http:.../train"
url_predict = "http:.../predict"

# specify local path to dataset file 
dataset_path = r"..." 

# upload dataset to server
with open(dataset_path, 'rb') as f:
    files = {'file': (dataset_path, f, 'text/csv')}
    response_upload = requests.post(url_upload, files = files)

# model training
response_train = requests.post(url_train)

# define sample input
sample_input = {
    'Temperature' : 70,
    'Run_Time' : 300
}

# convert python dictionary to JSON string
input_json = json.dumps(sample_input)

# request prediction from the trained model
response = requests.post(url_predict, data = input_json)

print(response.text)
