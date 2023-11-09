import time


class Timer:
    def __init__(self, logger):
        self.logger = logger
        self.start_time = time.time()
        self.last_checkpoint = self.start_time

    def checkpoint(self, name):
        now = time.time()
        elapsed = now - self.last_checkpoint
        self.last_checkpoint = now
        self.logger.info(f"{name} took: {elapsed:.3f} seconds")

    def total(self):
        total_elapsed = time.time() - self.start_time
        self.logger.info(f"Total time: {total_elapsed:.3f} seconds")
