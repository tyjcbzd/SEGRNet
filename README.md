# SEGRNet：GENERALIZED SYNERGISTIC EDGE-GUIDED GRAPH REASONING NETWORK FOR BIOMEDICAL IMAGE SEGMENTATION
This is the official implementation code for article named "GENERALIZED SYNERGISTIC EDGE-GUIDED GRAPH REASONING NETWORK FOR BIOMEDICAL IMAGE SEGMENTATION".

## Overview

Biomedical image segmentation plays a vital role in computer-aided diagnosis and treatment planning. However, existing methods often struggle with modeling complex anatomical structures and capturing long-range dependencies. To address these limitations, we propose a generalized Synergistic Edge-Guided Graph Reasoning Network (SEGRNet) that integrates convolutional feature extraction with graph-based global reasoning. The model projects pixel-level region and edge features into a graph domain, enabling adaptive interaction between local and global features via a graph convolutional network. After reasoning, enhanced features are mapped back for refined segmentation. Experiments conducted on three public datasets including BUSI, LGG and CHAOS outperforms state-of-the-art models in terms of dice coefficient, mean intersection over union and structural similarity. These results confirm the effectiveness and generalization ability of the proposed method across various medical imaging scenarios, making it suitable for future clinical applications.


This paper introduces a novel Edge-guided and Hierarchical Aggregation Network (EHANet) which excels at capturing rich contextual information and preserving fine spatial details, addressing the critical issues of inaccurate mask edges and detail loss prevalent in current segmentation models. The Inter-layer Edge-aware Module (IEM) enhances edge prediction accuracy by fusing early encoder layers, ensuring precise edge delineation. The Efficient Fusion Attention Module (EFA) adaptively emphasizes critical spatial and channel features while filtering out redundancies, enhancing the model's perception and representation capabilities. The Adaptive Hierarchical Feature Aggregation Module (AHFA) module optimizes feature fusion within the decoder, maintaining essential information and improving reconstruction fidelity through hierarchical processing. 


![Image 1](imgs/Overview.png)


### Synergistic Edge-Guided Graph Reasoning (SEGR) module
![Image 2](images/AHFA.png)


## Experimental results


## :black_nib: For citation


:exclamation: :eyes: **The codes can not be used for commercial purposes!!!**

