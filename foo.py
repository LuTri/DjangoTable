import matplotlib.pyplot as plt
import json

with open('mixed.json', 'r') as fp:
    data = json.load(fp)
    
# plot
fig, ax = plt.subplots()  

x_vals = []
y_vals = []

avg = None

for x, y in enumerate(data):
    x_vals.append(x)
    y_vals.append(y)
    
    if avg is None:
        avg = y
    else:
        avg = (avg + y) / 2.0
    
ax.plot(x_vals, y_vals, linewidth=1.0, label=f'average: {avg:.6f} (== {1/avg}FPS)')
ax.legend()

ax.set(ylim=(0, max(y_vals)))

plt.show()