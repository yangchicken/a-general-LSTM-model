import logging
import os

def setup_logger(log_folder):
    """
    Configures a logger to record information during the training process.
    Args:
    log_file (str): The path to the log file; defaults to saving to a log file.
    Returns:
    logger: The configured logger.
    """
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    log_file = os.path.join(log_folder, "training.log")

    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.INFO)
    
    # Create a file handler and a console handler.
    file_handler = logging.FileHandler(log_file)
    
    # Set the log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add processor
    logger.addHandler(file_handler)
    return logger