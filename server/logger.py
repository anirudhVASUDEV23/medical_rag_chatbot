import logging 

def setup_logger(name="MedicalAssistant"):
    logger=logging.getLogger(name)
    logger.setLevel(logging.DEBUG)##all logs from debug level and above will be printed


    ch=logging.StreamHandler()##will print logs to console
    ch.setLevel(logging.DEBUG)##all logs from debug level and above will be printed


    formatter=logging.Formatter("[%(asctime)s] - [%(name)s] - [%(levelname)s] - [%(message)s]")
    ch.setFormatter(formatter)

    if not logger.hasHandlers():##if logger has no handlers  only then add console handler
        logger.addHandler(ch)

    return logger


logger=setup_logger()


