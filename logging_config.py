import logging
import os
from logging.handlers import TimedRotatingFileHandler

# Ensure the logs directory exists
LOG_DIRECTORY = "logs"
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# Configure logging
log_file_path = os.path.join(LOG_DIRECTORY, "app.log")
handler = TimedRotatingFileHandler(log_file_path, when="midnight", interval=1, backupCount=7)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.NOTSET)

# Avoid adding the handler multiple times
if not logger.hasHandlers():
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
