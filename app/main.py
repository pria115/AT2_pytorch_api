from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import torch.nn as nn
import torch
import pandas as pd
from pandas.core.common import maybe_box_datetimelike
from sklearn.preprocessing import StandardScaler
import category_encoders as ce

app = FastAPI()

import os
import sys
sys.path.append(os.path.abspath('.'))


pytorch = torch.load('../models/pytorch_beer.pt', encoding='ascii')


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

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')  
    return device


@app.get("/beer/style/segmentation")
def predict(brewery_name: str,	review_aroma: int, review_appearance: int, review_palate: int, review_taste: int):
    features = format_features(brewery_name, review_aroma, review_appearance, review_palate, review_taste)
    obs = pd.DataFrame(features)

    
    sc = StandardScaler()
    num_cols = ['review_palate', 'review_aroma', 'review_taste', 'review_appearance']
    obs[num_cols] = sc.fit_transform(obs[num_cols])
    
    cat_cols = ['brewery_name']
    encoder = ce.BinaryEncoder(cols=cat_cols)
    obs = encoder.fit_transform(obs)

    df_tensor= torch.Tensor(np.array(obs)).to(device)

    device = get_device()
    df_tensor= torch.Tensor(np.array(obs)).to(device)

    output = pytorch(df_tensor)

    pred = pytorch.predict(df_tensor)

    return JSONResponse(pred.tolist())
