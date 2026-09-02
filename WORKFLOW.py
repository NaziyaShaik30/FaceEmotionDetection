import matplotlib.pyplot as plt
import networkx as nx
# Create a directed graph
G = nx.DiGraph()
# Define workflow steps
steps = [
    "Retrieve Dataset", "Extract Labels", "Load Audio Files", "Extract MFCC Features",
    "Check Sequence Length", "Pad or Truncate", "Store Features (X) & Labels (y)",
    "Convert to NumPy Arrays", "Done"
]
# Add edges to define the workflow
edges = [
    ("Retrieve Dataset", "Extract Labels"),
    ("Extract Labels", "Load Audio Files"),
    ("Load Audio Files", "Extract MFCC Features"),
    ("Extract MFCC Features", "Check Sequence Length"),
    ("Check Sequence Length", "Pad or Truncate"),
    ("Pad or Truncate", "Store Features (X) & Labels (y)"),
    ("Store Features (X) & Labels (y)", "Convert to NumPy Arrays"),
    ("Convert to NumPy Arrays", "Done")
]

G.add_nodes_from(steps)
G.add_edges_from(edges)

# Define positions using a hierarchical layout
pos = nx.shell_layout(G)

# Define node colors based on categories
node_colors = ["lightgreen" if step in ["Retrieve Dataset", "Extract Labels"] else
               "lightblue" if step in ["Load Audio Files", "Extract MFCC Features", "Check Sequence Length", "Pad or Truncate"]
               else "lightcoral" for step in steps]

# Draw the nodes with different colors
plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color="black",
        node_size=3000, font_size=10, font_weight="bold", arrows=True)

plt.title("Audio Preprocessing Workflow", fontsize=14)
plt.show()
