

class InstanceSingleton(type):
    def __init__(cls, name, bases, dict):
        super(InstanceSingleton, cls).__init__(name, bases, dict)
        cls.instances = {}

    def __call__(cls, *args):
        try:
            cls.instances[args]
        except KeyError:
            return super(InstanceSingleton, cls).__call__(*args)

    def __new__(cls, name, bases, dict):
        dict['__deepcopy__'] = dict['__copy__'] = lambda self, *args: self
        return super(InstanceSingleton, cls).__new__(cls, name, bases, dict)
