class Example:
    def __init__(self, value):
        self.value = value

    @classmethod
    @some_decorate(foo="bar")
    def create(cls, value):
        return cls(value)

    @classmethod
    def create(cls, value):
        return cls(value)
    
    async def fetch(self):
        return self.value


def helper():
    def nested():
        return 42

    return nested()