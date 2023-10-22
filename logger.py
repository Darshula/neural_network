import datetime
import sys


class Logger:
    def __init__(self, file: str | None = None, error_file: str | None = None,
                 output_level: int = 3) -> None:
        if output_level > 3 or output_level < 1:
            raise ValueError('`output_level` must lie between 1 and 3')
        self.output_level = output_level
        if file is not None:
            self.file = open(file, 'w+')
            if error_file is not None:
                self.error_file = open(error_file)
            else:
                self.error_file = self.file
        else:
            self.file = sys.stdout

    def log_info(self, *args):
        if self.output_level == 3:
            print(datetime.datetime.now().time(), Strings.info,
                  * args,
                  file=sys.stdout if self.file is None
                  else self.file)

    def log_warn(self, *args):
        if self.output_level > 1:
            print(datetime.datetime.now().time(), Strings.warn,
                  * args,
                  file=sys.stdout if self.file is None
                  else self.file)

    def log_error(self, *args):
        if self.output_level > 0:
            print(datetime.datetime.now().time(), Strings.error,
                  *args,
                  file=sys.stderr if self.error_file is None
                  else self.error_file)


class Strings:
    info = '[INFO]: '
    warn = '[WARN]: '
    error = '[ERROR]: '
