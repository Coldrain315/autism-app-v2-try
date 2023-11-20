from fastapi import FastAPI, UploadFile, File
from starlette.middleware.cors import CORSMiddleware
import asyncio
from api.tracker import TrackerService
from api import tracker
import pandas as pd
import os
from fastapi import File
from tempfile import TemporaryDirectory
from api import model

# Initialize Tracker Service
tracker_service = TrackerService()
bucket_name = os.environ["GCS_BUCKET_DATA"]
local_txt_path = "/persistent/txt"
local_experiments_path = "/persistent/experiments"

# Setup FastAPI app
app = FastAPI(title="API Server", description="API Server", version="v1")

# Enable CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    print("Startup tasks")
    # Start the tracker service
    asyncio.create_task(tracker_service.track())


# Routes
@app.get("/")
async def get_index():
    return {"message": "Welcome to the API Service. APCOMP215"}

@app.post("/upload")
async def upload_txt(file: UploadFile = File(...)):
    print("Received file:", "test.txt")
    # Save the image to a temporary directory
    with TemporaryDirectory() as text_dir:
        text_path = os.path.join(text_dir, "test.txt")
        with open(text_path, "wb") as output:
            output.write(await file.read())
            print("Load text in " + "text_path")
        model.load_text_from_path(text_path)
        # Load and preprocess the image
        # model.load_preprocess_image(image_path)
    return {"message": "data loaded and preprocessed successfully"}

@app.post("/load")
async def load_txt():
    tracker.download_text()
    model.load_text_from_path("/persistent/txt/text.txt")
    return {"message": "data loaded and preprocessed successfully"}
# @app.get("/experiments")
# def experiments_fetch():
#     # Fetch experiments
#     df = pd.read_csv("/persistent/experiments/experiments.csv")
#
#     df["id"] = df.index
#     df = df.fillna("")
#
#     return df.to_dict("records")
#
#
# @app.get("/best_model")
# async def get_best_model():
#     model.check_model_change()
#     if model.best_model is None:
#         return {"message": "No model available to serve"}
#     else:
#         return {
#             "message": "Current model being served:" + model.best_model["model_name"],
#             "model_details": model.best_model,
#         }
#
#
# @app.post("/predict")
# async def predict(file: bytes = File(...)):
#     print("predict file:", len(file), type(file))
#
#     self_host_model = True
#
#     # Save the image
#     with TemporaryDirectory() as image_dir:
#         image_path = os.path.join(image_dir, "test.png")
#         with open(image_path, "wb") as output:
#             output.write(file)
#
#         # Make prediction
#         prediction_results = {}
#         if self_host_model:
#             prediction_results = model.make_prediction(image_path)
#         else:
#             prediction_results = model.make_prediction_vertexai(image_path)
#
#     print(prediction_results)
#     return prediction_results
