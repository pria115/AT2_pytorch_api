from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import torch.nn as nn
import torch
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath('.'))

from src.models.pytorch import PytorchMultiClass

class PytorchMultiClass(nn.Module):
    def __init__(self, num_features):
        super(PytorchMultiClass, self).__init__()
        
        self.layer_1 = nn.Linear(num_features, 32)
        self.layer_out = nn.Linear(32, 104)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.dropout(F.relu(self.layer_1(x)), training=self.training)
        x = self.layer_out(x)
        return self.softmax(x)

app = FastAPI()

pytorch = torch.load('../models/pytorch_beer_style.pt', encoding='ascii')

@app.get("/")
def read_root():
    return {"Hello": "BeerFans"}

@app.get('/health', status_code=200)
def healthcheck():
    return 'Get ready to taste the best beer!!'


def format_features(brewery_name: str,	review_aroma: int, review_appearance: int, review_palate: int, review_taste: int):
  return {
        'Brewery': [brewery_name],
        'Aroma': [review_aroma],
        'Appearance': [review_appearance],
        'Palate': [review_palate],
	'Taste': [review_taste]
    }


@app.get("/mall/customers/segmentation")
def predict(brewery_name: str,	review_aroma: int, review_appearance: int, review_palate: int, review_taste: int):
    features = format_features(brewery_name, review_aroma, review_appearance, review_palate, review_taste)
    obs = pd.DataFrame(features)
    pred = pytorch_beer_style.predict(obs)
    return JSONResponse(pred.tolist())
