# kaggle-Google-AI-Comp
### Goal of the Competition

Alice is an AI model developer, but some of the models her team developed run very slow. She recently discovered compiler's configurations that change the way the compiler compiles and optimizes the models, and hence make the models run faster (or slower)! 

Train a machine learning model based on the runtime data provided to you in the training dataset and further predict the runtime of graphs and configurations in the test dataset.

### Context
An AI model can be represented as a graph, where a node is a tensor operation (e.g. matrix multiplication, convolution, etc), and an edge represents a tensor. A compilation configuration controls how the compiler transforms the graph for a specific optimization pass. In particular, Alice can control two types of configurations/optimizations:

·A layout configuration control how tensors in the graph are laid out in the physical memory, by specifying the dimension order of each input and output of an operation node.

·A tile configuration controls the tile size of each fused subgraph.
![image](https://github.com/kaamava/kaggle-Google-AI-Comp/assets/106901273/a857a8b7-5f01-486d-b7a7-7522957c0b00)
![image](https://github.com/kaamava/kaggle-Google-AI-Comp/assets/106901273/a84c79da-e9c1-4f95-86af-de7426cd070f)


### Project Introduction
We used GCN and Simple MLP in our project.

1.Download official data to the input folder.

2.Run five files sequentially: tile.ipynb, layout_default.ipynb, layout_random.ipynb, nlp_default.ipynb, nlp_random.ipynb.

3.Execute combine.ipynb to obtain submission.csv.

Note:Significant memory is required for execution; it is recommended to have 128GB RAM and 40GB GPU memory

### Our Advantages
1.This competition is based on the structure and operational time data of various deep learning models to accurately infer the consumption time of each node in the models, thereby optimizing and adjusting the model structure. The competition includes five regression prediction problems. To evaluate the accuracy of the predictions, the competition adopts two evaluation metrics: topK accuracy and correlation coefficient. 

2.Considering the type of data in this competition, we chose to use a Graph Convolutional Network (GCN) model to address this challenge. Initially, we perform embedding encoding on the operation codes (node_opcode) and node configuration features (node_config_feat) to capture the latent representations of these discrete attributes. These encodings, combined with other node features (node_feat), serve as inputs to the nodes in the GCN model. In terms of model architecture, we use GCN convolution layers (GCNConv) to process the graph structure, facilitating the flow and integration of information between nodes. For the training objective of the model, we utilize the ListMLE loss function, a common choice in list-wise learning, suitable for addressing sorting or priority-related problems. Finally, we opt to train the model using the Adam optimizer. 

3.The performance of a single model is often limited by its structure and parameters. To further enhance the accuracy of predictions, we decided to employ model fusion on the aforementioned model. Specifically, we combined the prediction results of multiple models through weighted fusion, then sorted and output these results. This strategy effectively reduces the model's bias and enhances its robustness. 

4.In the competition, we made full use of the allotted time, managing the competition's pace effectively; we engaged in active discussions, devised a reasonable model iteration plan for the competition, and organized brainstorming sessions to advance the competition's progress. Meanwhile, everyone embarked on the model development work, ensuring to enhance the prediction accuracy as much as possible. 

