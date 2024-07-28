import logging
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from paligemma import PaliGemma
from logging_config import logger  # Import the logger

# Ensure the uploads directory exists
UPLOAD_DIRECTORY = "uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

app = FastAPI()
model = PaliGemma()

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
