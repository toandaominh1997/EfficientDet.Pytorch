

def sum():
    a = 10
    b = 20
    for name in dir():
        if not name.startswith('_'):
            del locals()[name]

sum()
