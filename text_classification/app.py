import os
import sys

sys.path.append(".")

from fastapi import FastAPI, Path
from fastapi.responses import RedirectResponse

from http import HTTPStatus
from pydantic import BaseModel

import wandb
import json

from text_classification import config, data, predict, utils


app = FastAPI(
    title="text-classification",
    description="",
    version="1.0.0",
)


# Get best run
best_run = utils.get_best_run(project="mahjouri-saamahn/mwml-httynn-app_v2",
                              metric="test_loss", objective="minimize")

# Load best run (if needed)
best_run_dir = utils.load_run(run=best_run)

# Get run components for prediction
args, model, X_tokenizer, y_tokenizer = predict.get_run_components(
    run_dir=best_run_dir)



@utils.construct_response
@app.get("/")
async def _index():
    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
        'data': {}
    }
    config.logger.info(json.dumps(response, indent=2))

    return response


@app.get("/experiments")
async def _experiments():
    return RedirectResponse("https://app.wandb.ai/mahjouri-saamahn/mwml-httynn-app_v2")


class PredictPayload(BaseModel):
    experiment_id: str = 'latest'
    inputs: list = [{"text": ""}]


@utils.construct_response
@app.post("/predict")
async def _predict(payload: PredictPayload):
    prediction = predict.predict(inputs=payload.inputs, args=args, model=model,
                                 X_tokenizer=X_tokenizer, y_tokenizer=y_tokenizer)
    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
        'data': {"prediction": prediction}
    }
    config.logger.info(json.dumps(response, indent=2))
    
    return response