# logging_config.py

import logging
from torch import distributed as dist

# levels are DEBUG, INFO, WARNING, ERROR, CRITICAL

def configure_logging(enable_logging=False):

    logging.getLogger('asyncio').setLevel(logging.WARNING) # corner case, comes with python 3, by default logs at level DEBUG

    if dist.is_initialized() and dist.get_rank() != 0:
        # If rank is not 0, disable logging by setting a high log level
        logging.basicConfig(level=logging.WARNING, force=True) 
        return

    if enable_logging:
        logging.basicConfig(
            level=logging.DEBUG,  # Set the minimum log level to DEBUG
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),  # Log to the console
                #logging.FileHandler("app.log")  # Also log to a file
            ],
            force=True
        )
    else:
        logging.basicConfig(
            level=logging.INFO,  # Set the minimum log level to INFO (next level is warning)
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),  # Log to the console
                #logging.FileHandler("app.log")  # Also log to a file
            ],
            force=True
        )
