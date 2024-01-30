# TODO
class Error(Exception):
    pass


class ExampleError(Error):
    def __init__(self, msg):
        self.msg = msg