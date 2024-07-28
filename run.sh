#!/bin/bash

# Activate the virtual environment
source paligemma-env/bin/activate 

# Run the FastAPI application with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --log-config logging_config.py