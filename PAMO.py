from fastapi import FastAPI, File, UploadFile
import json
import pandas as pd
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
import pickle

# initialize FastAPI app and global variable for storing the model
app = FastAPI()
model = None

# dataset upload endpoint
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = pd.read_csv(file.file)
    data.to_csv("data.csv", index = False)
    return {"message" : "Data uploaded successfully"}

# model training endpoint
@app.post("/train")
async def train():
    global model
    data = pd.read_csv("data.csv")
    X = data[['Temperature', 'Run_Time']] # inputs
    y = data['Downtime_Flag'] # target
    model = LogisticRegression()
    model.fit(X, y) # train model
    pickle.dump(model, open("model.pkl", "wb"))
    return {"message" : "Model trained successfully"}

# define input data structure for prediction using pydantic
class InputsForPrediction(BaseModel):
    Temperature: float
    Run_Time: float

# prediction endpoint
@app.post("/predict")
async def predict(input_data: InputsForPrediction):
    global model
    if not model:
        model = pickle.load(open("model.pkl", "rb")) # load model from file, if not loaded

    # convert input data to JSON format
    inputs = input_data.model_dump_json()
    inputs_dict = json.loads(inputs)

    # set up inputs for prediction
    temperature = inputs_dict['Temperature']
    run_time = inputs_dict['Run_Time']

    input_list = [temperature, run_time]

    prediction  = model.predict([input_list])[0] # prediction
    confidence = max(model.predict_proba([input_list])[0]) # confidence of prediction

    # return prediction result and confidence as JSON
    return {"Downtime" : "Yes" if prediction == 1 else "No", "Confidence" : confidence} 