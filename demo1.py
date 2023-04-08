import matplotlib.pyplot as plt

fix, ax = plt.subplots()
ax.scatter(x=1, y=2, marker="x", label="A") # type: ignore
ax.scatter(x=2, y=1, marker="+", label="B") # type: ignore
ax.plot([1, 2, 3], [1, 2, 4], marker="^", label="plot")
plt.legend()
plt.show()