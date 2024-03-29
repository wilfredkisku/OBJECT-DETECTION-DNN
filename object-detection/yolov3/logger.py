import logging
#from logging import Filter
#from logging.handlers import QueueHandler, QueueListener
#from torch.multiprocessing import Queue

def build_basic_logger(log_file_path: str, set_level=2):
    output_file_log_handler = logging.FileHandler(filename=str(log_file_path))
    formatter = logging.Formatter('%(asctime)s | %(message)s', '%Y-%m-%d %H:%M:%S')
    output_file_log_handler.setFormatter(formatter)

    logger_levels = [
        logging.DEBUG, # set_level = 0
        logging.INFO, # set_level = 1
        logging.WARNING, # set_level = 2
        logging.ERROR, # set_level = 3
        logging.CRITICAL # set_level = 4
    ]

    logger = logging.getLogger()
    logger.setLevel(level=logger_levels[set_level])
    logger.addHandler(output_file_log_handler)
    return logger

if __name__ == "__main__":

    log_file_path = "/home/wilfred/Desktop/object-detection/yolov3/log/logger_test.log"
