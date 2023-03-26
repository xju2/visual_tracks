import logging

def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes python command line logger."""

    logger = logging.getLogger(name)

    # # this ensures all logging levels get marked with the rank zero decorator
    # # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    # logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    # for level in logging_levels:
    #     setattr(logger, level, getattr(logger, level))

    return logger
