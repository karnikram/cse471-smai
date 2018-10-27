def gradient(x):
    return 2*x

# Initialize
x = 10
v = 0

# Update equations
v = v*0.5 - 0.1 * gradient(x)
x_new = x + v

# Iteration count
i = 1

while(abs(x_new - x) >= 0.0001):
    x = x_new
    print(str(i) + ":" + str(x_new))
    v= v*0.5 -0.1 * gradient(x)
    x_new = x + v
    i+=1


# ordinary gradient descent update
# x_new = x - 0.1 * gradient(x)
