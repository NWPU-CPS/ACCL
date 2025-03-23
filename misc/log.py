import os
from datetime import datetime
from inspect import currentframe, getframeinfo


class MeowLogger(object):
    def __init__(self):
        self.logf = None

    def __del__(self):
        if self.logf is not None:
            self.logf.close()

    def __header(self, pid):
        now = datetime.now()
        frameInfo = getframeinfo(currentframe().f_back.f_back)
        if pid:
            return "[\033[90m{}|\033[0m{}:{}|{}] ".format(now.strftime("%Y-%m-%dT%H:%M:%S.%f"), os.path.basename(frameInfo.filename), frameInfo.lineno, os.getpid())
        return "[\033[90m{}|\033[0m{}:{}] ".format(now.strftime("%Y-%m-%dT%H:%M:%S.%f"), os.path.basename(frameInfo.filename), frameInfo.lineno)

    def setLogFile(self, filename):
        if self.logf is not None:
            self.logf.close()
        self.logf = open(filename, "w")

    def log(self, content, muted=False):
        if muted:
            return
        if self.logf is not None:
            self.logf.write(content + "\n")
            self.logf.flush()
            return
        print(content)

    def inf(self, line, pid=False, muted=False):
        self.log(self.__header(pid) + line, muted)

    def grey(self, line, pid=False, muted=False):
        self.log("{}\033[90m{}\033[0m".format(self.__header(pid), line), muted)

    def red(self, line, pid=False, muted=False):
        self.log("{}\033[91m{}\033[0m".format(self.__header(pid), line), muted)

    def green(self, line, pid=False, muted=False):
        self.log("{}\033[92m{}\033[0m".format(self.__header(pid), line), muted)

    def yellow(self, line, pid=False, muted=False):
        self.log("{}\033[93m{}\033[0m".format(self.__header(pid), line), muted)

    def blue(self, line, pid=False, muted=False):
        self.log("{}\033[94m{}\033[0m".format(self.__header(pid), line), muted)

    def pink(self, line, pid=False, muted=False):
        self.log("{}\033[95m{}\033[0m".format(self.__header(pid), line), muted)

    def cyan(self, line, pid=False, muted=False):
        self.log("{}\033[96m{}\033[0m".format(self.__header(pid), line), muted)


log = MeowLogger()
