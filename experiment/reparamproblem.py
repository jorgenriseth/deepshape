from argparse import ArgumentParser

class ReparametrizationParser(ArgumentParser):
    def __init__(self):
        super().__init__(description="")
        self.parse_arg()
