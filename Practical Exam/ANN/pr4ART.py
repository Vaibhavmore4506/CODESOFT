import numpy as np

class ART:
    def __init__(self, input_size, vigilance=0.7):
        self.input_size = input_size
        self.vigilance = vigilance
        self.weights = []  # List to store category weights

    def train(self, inputs):
        for x in inputs:
            matched = False
            for i, w in enumerate(self.weights):
                similarity = np.sum(np.minimum(x, w)) / np.sum(x)
                print(f"Checking similarity with Category {i+1}: {similarity:.2f}")
                
                if similarity >= self.vigilance:
                    print(f"Pattern matched with Category {i+1}. Updating weights.")
                    self.weights[i] = np.minimum(x, w)  # Update existing category
                    matched = True
                    break

            if not matched:
                print("No match found. Creating a new category.")
                self.weights.append(x)  # Create new category

    def test(self, x):
        print("\nTesting new pattern:", x)
        for i, w in enumerate(self.weights):
            similarity = np.sum(np.minimum(x, w)) / np.sum(x)
            print(f"Similarity with Category {i+1}: {similarity:.2f}")
            if similarity >= self.vigilance:
                return f"Pattern belongs to Category {i+1}"
        return "New category required"

# Define input patterns
inputs = np.array([
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 1, 1]
])

# Create ART network with vigilance 0.7
art_network = ART(input_size=4, vigilance=0.7)

# Train the network
art_network.train(inputs)

# Test with a new pattern
new_pattern = np.array([1, 1, 0, 1])
print(art_network.test(new_pattern))
