# Import Dataset
import math
import numpy as np

# Read Dataset
# Points Dataset
xp = [0.1, 0.15, 0.08, 0.16, 0.2, 0.25, 0.24, 0.3]
yp = [0.6, 0.71, 0.9, 0.85, 0.3, 0.5, 0.1, 0.2]
l = len(xp)

# Centroid Dataset C1
C1x = 0.1
C1y = 0.6

# Centroid Dataset C2
C2x = 0.3
C2y = 0.2

# Create Cluster m1
m1x = []
m1y = []
m1x.append(C1x)
m1y.append(C1y)

# Create Cluster m2
m2x = []
m2y = []
m2x.append(C2x)
m2y.append(C2y)

# Find New Centroid & Update Population / Insert Point into Respective Cluster
for i in range(1, l):   # starting from 1 because first point is already taken
    dist1 = math.sqrt((xp[i] - C1x)**2 + (yp[i] - C1y)**2)
    dist2 = math.sqrt((xp[i] - C2x)**2 + (yp[i] - C2y)**2)

    # Update Population
    if dist1 < dist2:
        m1x.append(xp[i])
        m1y.append(yp[i])
    else:
        m2x.append(xp[i])
        m2y.append(yp[i])

# Calculating length of new Population in m1 and m2
l1 = len(m1x)
l2 = len(m2x)

# Finding P6 (xfind=0.25, yfind=0.5) belong to which cluster
xfind = 0.25
yfind = 0.5

found = False
for i in range(l1):
    if (xfind == m1x[i]) and (yfind == m1y[i]):
        print("P6 Belongs to m1")
        found = True
        break

if not found:
    for i in range(l2):
        if (xfind == m2x[i]) and (yfind == m2y[i]):
            print("P6 Belongs to m2")
            break

# Showing Population Size of M1 & M2
print("\nPopulation of M1 Cluster:")
for i in range(l1):
    print(f"P{i+1} = [{m1x[i]}, {m1y[i]}]")
print(f"M1 Population Size = {l1}")

print("\nPopulation of M2 Cluster:")
for i in range(l2):
    print(f"P{i+1} = [{m2x[i]}, {m2y[i]}]")
print(f"M2 Population Size = {l2}")

# Finding Updated Centroid Value of C1 and C2
new_C1x = np.mean(m1x)
new_C1y = np.mean(m1y)
new_C2x = np.mean(m2x)
new_C2y = np.mean(m2y)

print(f"\nUpdated Centroid Value of C1 = [{new_C1x}, {new_C1y}]")
print(f"Updated Centroid Value of C2 = [{new_C2x}, {new_C2y}]")
