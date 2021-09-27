import time

class Logger:
    def __init__(self, interval):
        self.interval = interval
        self.start()
        
    def start(self):
        self.tic = time.time()
            
    def stop(self):
        self.toc = time.time()
        print()
        print(f'Finished training in {self.toc - self.tic:.5f}s')
        
    def log(self, it, value):
        if self.interval > 0 and it % self.interval == 0:
            print('[Iter %5d] loss: %.5f' % (it + 1, value))

            
class Silent(Logger):
    def __init__(self):
        super().__init__(0)