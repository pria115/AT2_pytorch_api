from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import torch.nn as nn
import torch
import pandas as pd

app = FastAPI()

import os
import sys
sys.path.append(os.path.abspath('.'))


pytorch = torch.load('../models/pytorch_beer1.pt', encoding='ascii')


@app.get("/")
def read_root():
    return {"Beer": "'O Clock"}

@app.get('/health', status_code=200)
def healthcheck():
    return 'Life is too short to drink bad beer!!'


def format_features(brewery_name: str,	review_aroma: int, review_appearance: int, review_palate: int, review_taste: int):
  return {
        'Brewery': [brewery_name],
        'Aroma': [review_aroma],
        'Appearance': [review_appearance],
        'Palate': [review_palate],
	'Taste': [review_taste]
    }


@app.get("/beer/style/segmentation")
def predict(brewery_name: str,	review_aroma: int, review_appearance: int, review_palate: int, review_taste: int):
    features = format_features(brewery_name, review_aroma, review_appearance, review_palate, review_taste)
    obs = pd.DataFrame(features)
    pred = pytorch_beer_style.predict(obs)
    return JSONResponse(pred.tolist())
