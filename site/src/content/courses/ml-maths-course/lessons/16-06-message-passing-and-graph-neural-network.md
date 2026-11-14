---
id: "16.06"
course: "ml-maths-course"
module: "16"
slug: "message-passing-and-graph-neural-networks"
title: "Message passing and graph neural networks"
keyIdeas: ["A message-passing layer updates each node from its own state and an aggregate of messages from its neighbours: $h_v' = \\phi\\big(h_v, \\bigoplus_{u\\in N(v)} \\psi(h_u, h_v, e_{uv})\\big)$.", "The graph convolutional network uses the normalised adjacency $\\hat{A} = \\tilde{D}^{-1/2}(A + I)\\tilde{D}^{-1/2}$, a low-pass filter over the graph: $H' = \\sigma(\\hat{A} H W)$.", "Stacking $L$ layers gives each node information from its $L$-hop neighbourhood; too many layers over-smooth, making all node states similar.", "Sum aggregation is more expressive than mean or max, and message-passing networks cannot distinguish graphs that the Weisfeiler-Lehman test cannot."]
question: "A GCN for node classification has 2 layers, each $H' = \\mathrm{ReLU}(\\hat{A} H W)$. (a) Which nodes can influence the prediction for node $v$? (b) You increase the depth to 16 layers and accuracy drops sharply, with node embeddings becoming nearly identical. Explain using the random-walk view of $\\hat{A}$. (c) Your graph has 50 million nodes with average degree 100. Why does full-neighbourhood mini-batch training of a 3-layer GCN become expensive, and what is the standard fix?"
answer: "(a) Nodes within 2 hops of $v$: each layer mixes in direct neighbours, so two layers reach neighbours of neighbours. (b) $\\hat{A}$ is a normalised averaging operator, closely related to a random-walk transition matrix with self-loops. Applying it 16 times is like running a walk for 16 steps, which approaches the stationary distribution: each node's representation becomes a degree-weighted average over its whole component, and differences between nodes vanish. This is over-smoothing. Fixes include fewer layers, residual or skip connections, and normalisation designed to keep node states apart. (c) The receptive field grows as roughly degree to the power of depth: a 3-hop neighbourhood can reach $100^3 = 10^6$ nodes per target node. Neighbour sampling (GraphSAGE-style, for example 10 to 25 sampled neighbours per hop), or clustering the graph into subgraphs, keeps each mini-batch bounded."
sources: [{"title": "Kipf and Welling, Semi-Supervised Classification with Graph Convolutional Networks (2016)", "url": "https://arxiv.org/abs/1609.02907", "note": "The GCN layer and its spectral motivation."}, {"title": "Gilmer et al., Neural Message Passing for Quantum Chemistry (2017)", "url": "https://arxiv.org/abs/1704.01212", "note": "The general message-passing framework."}, {"title": "Xu et al., How Powerful are Graph Neural Networks? (2018)", "url": "https://arxiv.org/abs/1810.00826", "note": "Expressiveness and the link to the Weisfeiler-Lehman test."}, {"title": "Li, Han and Wu, Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning (2018)", "url": "https://arxiv.org/abs/1801.07606", "note": "GCN as Laplacian smoothing, and over-smoothing."}]
practice: ["scaled-dot-product-attention", "masked-mean-pooling", "residual-block", "layer-normalization"]
tags: ["graph neural networks", "message passing", "graph convolution", "over-smoothing"]
tools: ["PyTorch"]
---

## The message-passing idea

Lesson 16.01 noted that $Ax$ sums neighbours' values. A graph neural network (GNN) learns what to send and how to combine it. In each layer, every node:

1. Computes a message from each neighbour, $m_{uv} = \psi(h_u, h_v, e_{uv})$, possibly using edge features.
2. Aggregates messages with a permutation-invariant function $\bigoplus$ (sum, mean, max or attention-weighted sum).
3. Updates its state, $h_v' = \phi(h_v, \text{aggregate})$, with a small neural network.

The same functions are shared across all nodes, so the model works on graphs of any size and shape, and it does not depend on how nodes are numbered.

~~~dg-flow One message-passing layer for node v.
Neighbour states h_u -(message function)-> Messages -(sum, mean, max or attention)-> *Aggregate -(update with h_v)-> New state h_v'
~~~

## Graph convolutional networks

The GCN layer is the simplest widely used instance:
$H' = \sigma(\hat{A} H W)$, with $\hat{A} = \tilde{D}^{-1/2}(A + I)\tilde{D}^{-1/2}$,
where $\tilde{D}$ is the degree matrix of $A + I$. Adding self-loops keeps each node's own features; the symmetric normalisation stops high-degree nodes from producing huge sums.

The spectral connection: $\hat{A} = I - \tilde{L}_{\text{sym}}$, where $\tilde{L}_{\text{sym}}$ is the normalised Laplacian of the graph with self-loops. In the Laplacian eigenbasis (lesson 16.02), multiplying by $\hat{A}$ scales each frequency component by $1 - \lambda$, which keeps smooth components and damps rapidly varying ones. A GCN layer is a fixed low-pass filter followed by a learned linear map and a non-linearity. This is why GCNs work well when neighbours tend to share labels (homophily) and poorly when they do not.

## Depth, receptive field and over-smoothing

After $L$ layers, node $v$'s state depends on its $L$-hop neighbourhood. More layers see further but also average more. Repeated multiplication by $\hat{A}$ behaves like a random walk run for many steps (lesson 16.04): states converge toward a degree-weighted average over each connected component, so nodes become indistinguishable. Most GNNs therefore use 2 to 4 layers, with residual connections, normalisation, or jumping-knowledge schemes that combine outputs from several depths.

<div class="ex">
<div class="lab">Worked example</div>

**deeper was worse**

A team classifying papers in a citation graph found accuracy of about 81% with 2 GCN layers, falling to about 70% at 8 layers and lower beyond. The cosine similarity between embeddings of randomly chosen nodes rose from 0.3 to 0.95 as depth increased. Adding residual connections and using 3 layers with a wider hidden size recovered and slightly exceeded the 2-layer result.

</div>

## Aggregators and expressiveness

- Mean aggregation forgets how many neighbours a node has; max keeps only the strongest signal; sum keeps both counts and contents.
- Xu and colleagues showed that message-passing GNNs are at most as powerful as the 1-dimensional Weisfeiler-Lehman graph isomorphism test, and that sum aggregation with an injective update (as in GIN) reaches that bound. Some simple structures, such as distinguishing two triangles from one hexagon when all nodes look alike, are beyond any such network without extra features.
- Attention (GAT) learns a weight per neighbour, so a node can focus on relevant neighbours. A transformer is a GNN on a complete graph with attention aggregation, with positional encodings to recover structure.

## Tasks and readouts

- Node classification: a head on each final node state.
- Link prediction: score a pair from its two node states, for example with a dot product, trained with sampled negative pairs.
- Graph classification (molecules, programs): pool all node states with sum, mean or attention into one vector (a readout), then classify.

## Scaling to large graphs

Full-batch training needs the whole graph in memory. Mini-batch training must include each target node's multi-hop neighbourhood, which grows roughly as degree to the power of depth. Remedies: sample a fixed number of neighbours per hop (GraphSAGE), partition the graph into clusters and train on subgraphs, or precompute $\hat{A}^k X$ once and train a simple model on the result.

## Common mistakes

- Leaking labels through the graph: in link prediction, the edges being predicted must be removed from the message-passing graph during training, or the model simply looks them up.
- Ignoring direction: citations and follows are directed; symmetrising them may help or hurt, so test both.
- Assuming homophily: on graphs where neighbours tend to differ (fraud rings linking to normal accounts, for example), plain GCN smoothing can hurt; keep a node's own features separate from the aggregate.

<div class="take">
<div class="lab">Key takeaway</div>

GNNs pass learned messages along edges and aggregate them with permutation-invariant functions; a GCN layer is low-pass filtering with the normalised adjacency, depth widens the receptive field but over-smooths, and sampling keeps large graphs tractable.

</div>
