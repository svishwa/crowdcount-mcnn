import time

class Timer(object):
    def __init__(self):
        self.tot_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.tot_time += self.diff
        self.calls += 1
        self.average_time = self.tot_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff
