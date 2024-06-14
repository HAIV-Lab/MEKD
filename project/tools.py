import os
import time


class Logger():
    def __init__(self, path: str):
        self.log_path = path
        self.log_dir = os.path.dirname(path)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    

    def info(self, content):
        time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        content = '[{0}] {1}\n'.format(time_stamp, content)

        with open(self.log_path, 'a') as f:
            f.write(content)