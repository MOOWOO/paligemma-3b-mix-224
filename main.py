import logging
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from paligemma import PaliGemma
from dotenv import load_dotenv
from logging_config import logger  # Import the logger
import re

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment variable
API_KEY = os.getenv("API_KEY")

# Ensure the uploads directory exists
UPLOAD_DIRECTORY = "uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

app = FastAPI()
#app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
model = PaliGemma()

def remove_special_characters(input_string):
    return re.sub(r'[^A-Za-z0-9]', '', input_string)

# Dependency to check for API key in the request header
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        logger.error("Invalid API Key")
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.post("/generate")
async def generate(task: str, image: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    try:
        logger.debug(f"Received request with task: {task}")
        
        # Read and save the uploaded image in the uploads directory
        image_path = os.path.join(UPLOAD_DIRECTORY, image.filename)
        contents = await image.read()
        with open(image_path, "wb") as f:
            f.write(contents)
        logger.debug(f"Image saved successfully as {image_path}")
        
        # Run the model
        output = model.run_raw_image(task, image_path)
        logger.debug(f"Model output: {output}")
        
        return {"output": output}
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/predict")
async def predict(task: str, image: str, api_key: str = Depends(verify_api_key)):
    try:
        logger.debug(f"Received request with task: {task} and image URL: {image}")

        # Run the model
        output = model.run(task, image)
        logger.debug(f"Model output: {output}")

        return {"output": output}
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/capcha")
async def generate(image: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    try:        
        # Read and save the uploaded image in the uploads directory
        image_path = os.path.join(UPLOAD_DIRECTORY, image.filename)
        contents = await image.read()
        with open(image_path, "wb") as f:
            f.write(contents)
        logger.debug(f"Image saved successfully as {image_path}")
        
        task = "caption(KOREAN letras, KOREAN LETTERS) el "
        output = model.run_raw_image(task, image_path)
        task = "'"+output +"' Las letras(LETTERS) escritas son?"
        output = model.run_raw_image(task, image_path)

        cleaned_string = remove_special_characters(output)
        logger.debug(f"Model output: {cleaned_string}")
        
        return {"output": cleaned_string}
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
