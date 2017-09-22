import codecs

def fun3(filename):
    f = codecs.open(filename,"r","utf-8")
    words = f.readline().strip()
    return len(words)