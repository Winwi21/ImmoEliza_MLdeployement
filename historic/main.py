from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])
pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "model.pkl")

model = joblib.load("gbr1_modele.pkl")

class InputData(BaseModel):
    type: str
    subtype: str
    bedroomCount: float
    bathroomCount: float
    province: str
    postCode: int
    habitableSurface: float
    buildingCondition: str
    facedeCount: float
    hasTerrace: int
    epcScore: str


app = FastAPI()

@app.post("/predict")
def predict(data: InputData):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)
    return {"predicted_price": prediction[0]}
