import pickle
import numpy as np

from tensorflow.keras.models import load_model

from modelCF import *
from utils import *

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

#------------------------------------------------------------------------------------------------------------------------------#

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


class Recommendation(BaseModel):
    userID: str


@app.post('/recsys/train')
def train():
    train_model_CF()
    '''
    train_model_CB()
    '''

    return 'Model trained successfully.'

@app.post('/recsys/recommend')
async def recommendItem(req: Recommendation):
    userID = req.userID

    # Load data, CF model, and its perforamnce
    uids, iids, df_train, df_test, df_neg, users, items = load_dataset()
    modelCF = load_model('model/modelCF.h5')
    with open('model/performance/hitrates_avg_CF.pkl', 'wb') as f:
        hitrates_avg = pickle.load(f)
    with open('model/performance/ndcgs_avg_CF.pkl', 'wb') as f:
        ndcgs_avg = pickle.load(f)
    
    try:
        # Predict ratings using CF model
        cf_works = True
        ratingsCF = predict_ratings_cf(
            user_idx = userID,
            items = items,
            model = modelCF
        )
        
        recommendations = get_top_k_items(ratingsCF, k = 10)
        average_ratings = np.mean([value for key, value in recommendations.items()])
    
        # If predicted rating avg. is less than 4 or HR@10 is less than 0.4 then CF model fails
        if (average_ratings >= 4) or (hitrates_avg >= 0.4) or (ndcgs_avg >= 0.3):
            recommendation_items = [key for key, value in recommendations.items()]
    
        # Else proceed to predict using CB model
        cf_works = False
        
        '''
        FILL THE CODE HERE FOR CB
        '''
        recommendation_items = []

        dct = {
            'status': 200,
            'message': "recommendation for user has been successfully get.",
            'data': {
                'user_id': userID,
                'recommendations': recommendation_items
                },
            'success': True,
            'error': None
        }

    except Exception as e:
        recommendation_items = []
        dct = {
            'status': 404,
            'message': "recommendation for user has failed.",
            'data': {
                'user_id': userID,
                'recommendations': recommendation_items
                },
            'success': False,
            'error': e
        }

    return dct