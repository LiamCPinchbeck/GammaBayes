
class MethodDict(dict):
    def __init__(self, *args, **kwargs):
        super(MethodDict, self).__init__(*args, **kwargs)
        # Set each key as an attribute
        for key, value in self.items():
            setattr(self, key, value)

    def __setitem__(self, key, value):
        super(MethodDict, self).__setitem__(key, value)
        # Update attribute when dictionary item is set
        setattr(self, key, value)

    def __delitem__(self, key):
        super(MethodDict, self).__delitem__(key)
        # Remove attribute when dictionary item is deleted
        delattr(self, key)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(f"'MethodDict' object has no attribute '{key}'") from e
