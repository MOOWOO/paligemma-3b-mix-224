import logging
import os
from logging.handlers import TimedRotatingFileHandler
from fastapi import FastAPI, File, UploadFile, HTTPException
from paligemma import PaliGemma
from PIL import Image

# Ensure the logs directory exists
LOG_DIRECTORY = "logs"
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# Configure logging
log_file_path = os.path.join(LOG_DIRECTORY, "app.log")
handler = TimedRotatingFileHandler(log_file_path, when="midnight", interval=1, backupCount=7)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

# Ensure the uploads directory exists
UPLOAD_DIRECTORY = "uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

app = FastAPI()
model = PaliGemma()

@app.post("/predict")
async def generate(task: str, image: str):
    try:
        logger.debug(f"Received request with task: {task}")

        # Run the model
        output = model.run(task, image)
        logger.debug(f"Model output: {output}")

        return {"output": output}
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
@app.post("/generate")
async def generate(task: str, image: UploadFile = File(...)):
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