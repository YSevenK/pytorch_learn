class Person:
    def __call__(self, name):
        print("__call__" + 'hello ' + name)

    def hello(self, name):
        print('hello ' + name)


Person = Person()
Person('zs')
Person.hello('ls')
