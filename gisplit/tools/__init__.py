
class classproperty(property):
    # Trick from https://stackoverflow.com/questions/128573/using-property-on-classmethods
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()  # pylint: disable=no-member
