def frange(start, stop, inc):
    i = start
    try:
        while i < stop:
            yield i
            i += inc
    except:
        pass

aj = xrange(3)
pa = xrange(2)

for i,j in zip(list(aj), list(pa)):
    print i,j



learning_rates = frange(1e-7, 1e-7 + 10.0 * 5e-5, 5e-5)
regularization_strengths = frange(5e4, 5e4 + 10 * 5e4, 5e4)

print list(learning_rates)
print list(regularization_strengths)

try:
    for lr in learning_rates:
        for rs in regularization_strengths:
            print "lr, rs", lr, rs
except:
        pass
