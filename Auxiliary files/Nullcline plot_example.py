import numpy as np
import matplotlib.pyplot as plt
mu = 3
c = -1
func1 = np.vectorize(lambda x: c*x)
func2 = np.vectorize(lambda x: x**2+mu)
x_values = np.linspace(-5, 5, 100)
y1_values = func1(x_values)
y2_values = func2(x_values)

plt.plot(x_values, y1_values, label="Nullcline y")
plt.plot(x_values, y2_values, label="Nullcline z")
plt.xlabel("y")
plt.ylabel("z")
plt.legend(loc=1)
plt.show()