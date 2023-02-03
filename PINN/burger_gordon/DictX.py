class DictX(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            print("===> error")
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<DictX ' + dict.__repr__(self) + '>'

#----------------------------------------------------------------------
class GlobDct(object):
    """ Singleton class to store global dictionary
        Args: None
        Return: DictX object
        Notes: There is only a single copy of the global dictionary.
    """
    def __new__(self):
        if not hasattr(self, 'instance'):
            self.instance = super(GlobDct, self).__new__(self)
            self.dct = DictX()
        return self.dct
