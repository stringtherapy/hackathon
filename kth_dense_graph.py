import k_dense
import networkx as nx
import functions
import matplotlib.pyplot as plt
from strawberryfields.apps import data, sample, subgraph, plot
import plotly

def plot_chosen_subgraph(graph,subgraph):
	fig = plot.graph(graph, subgraph)
	fig.show()

n = int(input('Enter number of nodes in the graph: '))
p = float(input('Enter probability: '))

graph = functions.random_graph_generation(n,p)
fig = plot.graph(graph)
fig.show()

k = int(input('Enter k value'))
dense = k_dense.indenify_kth_dense_subgraph(graph,k=k)

k_dense.plot_chosen_subgraph(graph,dense)