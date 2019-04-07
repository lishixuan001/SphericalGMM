import logging
import datetime
import sys
#import tensorflow as tf

def setup_logger(name):
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler('log/{}.log'.format(now), mode='w')
    handler.setFormatter(formatter)

    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)

    return logger

# class Logger(object):
#     """Tensorboard logger."""

#     def __init__(self, log_dir):
#         """Initialize summary writer."""
#         self.writer = tf.summary.FileWriter(log_dir)

#     def scalar_summary(self, tag, value, step):
#         """Add scalar summary."""
#         summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
#         self.writer.add_summary(summary, step)

#     def close(self):
#         self.writer.export_scalars_to_json("./all_scalars.json")
#         self.writer.close()