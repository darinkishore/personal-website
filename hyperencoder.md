# Hypencoder: Hypernetworks for Information Retrieval

**Julian Killingback**  
jkillingback@cs.umass.com  
University of Massachusetts Amherst  
Amherst, MA, USA

**Hansi Zeng**  
hzeng@cs.umass.edu  
University of Massachusetts Amherst  
Amherst, MA, USA

**Hamed Zamani**  
zamani@cs.umass.edu  
University of Massachusetts Amherst  
Amherst, MA, USA

## ABSTRACT

The vast majority of retrieval models depend on vector inner products to produce a relevance score between a query and a document. This naturally limits the expressiveness of the relevance score that can be employed. We propose a new paradigm, instead of producing a vector to represent the document, use a small neural network which acts as a learned relevance function. This small neural network takes in a representation of the document, in this paper we use a single vector, and produces a scalar relevance score. To produce the little neural network we use a hypernetwork, a network that produces the weights of other networks, as our query encoder or as we call it a Hypencoder. Experiments on in-domain search tasks show that Hypencoder is able to significantly outperform strong dense retrieval models and has higher metrics then reranking models and models an order of magnitude larger. Hypencoder is also shown to generalize well to out-of-domain search tasks. To assess the extent of Hypencoder’s capabilities, we evaluate on a set of hard retrieval tasks including tip-of-the-tongue retrieval and instruction-following retrieval tasks and find that the performance gap widens as the retrieval task becomes harder. To provide an understanding of the scalability of our model we implement an approximate search algorithm and show that our model is able to search 8.8M documents in under 60ms.

## 1 INTRODUCTION

Efficient neural retrieval models are based on a bi-encoder (or two tower) architecture, in which queries and documents are represented separately using either high-dimensional sparse [17, 18, 75] or relatively low-dimensional dense vectors [19, 32, 33, 76]. These models use simple and light-weight similarity functions, e.g., inner product or cosine similarity, to compute the relevance score for a given pair of query and document representations. We demonstrate theoretically that inner product similarity functions fundamentally limit the types of relevance that retrieval models can express. Specifically, we prove that there is always a set of relevant documents which cannot be perfectly retrieved regardless of the query vector and specific encoder model.

Motivated by this theoretical argument, we introduce a new category of retrieval models that can capture complex relationship between query and document representations. Building upon the hypernetwork literature in machine learning [22, 59, 66], we propose Hypencoder—a generic framework that learns a query-dependent multi-layer neural network as a similarity function that is applied to the document representations. In more detail, Hypencoder applies attention-based hypernetwork layers, called hyperhead layers, to the contextualized query embeddings output by a backbone transformer encoder. Each hyperhead layer produces the weight and bias matrices for a neural network layer in the query-dependent similarity network, called the q-net. The q-net is then applied to each document representation, which results in a scalar relevance score. We demonstrate that the Hypencoder framework can be optimized end-to-end and can be used for efficient retrieval from a large corpus. Specifically, we propose a graph-based greedy search algorithm that approximates exhaustive retrieval using Hypencoder while being substantially more efficient.

We conduct extensive experiments on a wide range of datasets to demonstrate the efficacy of Hypencoder. We demonstrate that our implementation of Hypencoder for single vector document representations outperforms competitive single vector dense and sparse retrieval models on MS MARCO [49] and TREC Deep Learning Track data [10, 13], and high difficulty retrieval tasks, such as TREC DL-Hard [43], TREC Tip-of-the-Tongue (TOT) Track [3], and the instruction following dataset FollowR [69]. Across these benchmarks Hypencoder demonstrates consistent performance gain across experiments. Note that using the proposed approximation approach, retrieval from MS MARCO [49] with approximately 8.8 million documents only takes an average of 56 milliseconds per query on a single NVIDIA 140S GPU.

A main advantage of hypernetworks in machine learning is their ability to learn generalizable representations. To demonstrate that Hypencoder also inherits this generalization quality, we evaluate our model under various domain adaptation settings: (1) adaptation to question answering datasets in biomedical and financial domains, and (2) adaptation to other retrieval tasks, including entity and argument retrieval, where Hypencoder again demonstrates superior performance compared to the baselines.

We believe that these performance gains are just the by-product of our main contributions; Hypencoder introduces a new way to think about what retrieval and relevance functions can be, it opens a new world of possibilities by bridging the gap between neural networks and retrieval similarity functions. We believe Hypencoder is especially important at this time when new demands for longer and more complex queries brought on by the widespread usage of large language models and it is our belief that Hypencoder represents an important step towards this goal. To help facilitate this goal we will open source all our code for training, retrieval, and evaluation. 

----

1 Available at https://github.com/jfkback/hypencoder-paper

# 2 RELATED WORK

## Vector Space Models

Vector-based models that use sparse vectors have existed for decades, with each index representing a term in the corpus vocabulary. Document-query similarity is computed using measures like L2 distance, inner product, or cosine similarity, with term weighting methods such as TF-IDF being a substantial focus to improve performance. With the emergence of deep neural networks, focus shifted to learning representations for queries and documents. SNRM by Zamani et al. [75] was the first deep learning model to retrieve documents from large corpora by learning latent sparse vectors. Following works leveraged pretrained transformer models like BERT [16] using dense vector representations [32]. Recent improvements have focused on training techniques including self-negative mining [52, 53, 73, 77], data augmentation [38, 53], distillation [26, 37, 38], corpus-pretraining [19, 30, 40, 71], negative-batch construction [28] and curriculum learning [39, 76]. Alternative approaches include ColBERT [33], which uses multiple dense vectors, and SPLADE [18], which results sparse representations using pretrained masked language models.

Though these methods vary substantially, they all share a fundamental commonality, that relevance is based on an inner product (or in some cases cosine similarity). We believe that this is a significant limitation of these methods and one which hampers the performance of these models on complex retrieval tasks. Our method circumvents this limitation by learning a query-dependent small neural network that is fast enough to run on the entire collection (or used in an approximate way; see Section 3.6 for details).

## Learned Relevance Models

Light-weight relevance models using neural networks have demonstrated improved retrieval performance compared to simple methods like inner products. Early iterations came in the form of learning-to-rank models [9, 18] which use query and document features to produce relevance scores for reranking. While these models traditionally used engineered features, more recent approaches adopted richer inputs. For instance, MatchPyramid [51] and KNRM [12] use similarity matrices between non-contextualized word embeddings, while Duet [47, 48] combines sparse and dense term features in a multi-layer perceptron. DRMM [74] utilized histogram features as input to neural networks for scoring. Since the advent of BERT [16], focus has shifted to making transformer models more efficient, such as PreTTR [42] which separately precomputes query and document hidden states. Recently, LITE [31] extended ColBERT’s similarity using column-wise and row-wise linear layers for scoring.

In the recommender systems community, learned similarity measures have been widely used [24, 25]. The common usage of neural scoring methods in recommendation has inspired research into efficient retrieval with more learned scoring signals. For instance, BFSG [62] supports efficient retrieval with arbitrary relevance functions by using a graph of item representations and a greedy search over nodes of the graph. A recent improvement on BFSG uses the scoring models gradient to prune directions that are unlikely to have relevant items [79]. Other works make use of queries from a query-item graph to produce more informative neighbors [61].

Our work differs from these works in one major way, we do not have a query representation and document representation thus our method requires no combination step, instead we produce a query-conditioned neural network for each query and directly apply this to the document representation. This approach can reduce the similarity network’s size and does not require choosing between inference speed and larger query representations. Furthermore the flexibility of our framework means we can replicate any existing learned relevance model as discussed in Section 3.7. On a broader note there has been surprisingly little work on neural based scoring for full-scale retrieval, especially in the modern era of transformer based encoders. We hope our work can be a useful foundation and proof-of-concept for future work in this area.

## Hypernetworks

Hypernetworks also known as hypernets are neural networks which produce the weights for other neural networks. The term was first used by Ha et al. [22] who demonstrated the effectiveness of hypernetworks to generate weights for LSTM networks. Since then, hypernetworks have been used in a variety of ways including neural architecture search [78], continual learning.

# Hyperencoder: Hypernetworks for Information Retrieval

[66], and few-shot learning [57, 59] to name a few. Generally, hypernetworks take a set of input embeddings that provide information about the type of task or network where the weights will be used. These embeddings are then projected to the significantly larger dimension of the weights of the "main" network. As the outputs of most hypernetworks are so large they themselves are often very simple such as a few feed-forward layers in order to keep computation feasible. Our case is unique in that our hypernetwork, the Hyperencoder, is much larger than the small scoring network which we call q-net (i.e. the "main" network). Additionally, to the best of our knowledge, this paper represents the first work to explore hypernetworks for first stage retrieval.

## 3 HYPENCODER

Neural ranking models can be generally categorized into early-interaction and late-interaction models [14, 20, 27, 33, 34, 50, 68]. Currently, the most common implementation of early-interaction models is in the form of cross-encoders (Figure 1 (second from the left)), where the query text q and document text d are concatenated (together with some predefined tokens or templates) and fed to a transformer network that learns a joint representation of query and document and finally produces a relevance score. The joint representation prevents these models from being able to precompute document representations, thus they cannot be used efficiently on large corpora [21, 46].

The most popular implementation of late-interaction models follows a bi-encoder (two tower) network architecture (Figure 1 (left)), where query and document representations are computed separately and a scoring function is used to estimate the relevance score. Formally, let Eq ∈ ℝ^k denote the representation for a query q consisting of an h-dimensional vectors. Similarly, Ed ∈ ℝ^k denotes the representation learned for document d consisting of m vectors of the same dimensionality. The relevance score between q and d is computed as follows:

\[ v(E_q, E_d) \]

where v: ℝ^m×k × ℝ^m×k → ℝ denotes the scoring function.

In order to take advantage of efficient indexing techniques, such as an inverted index in the case of sparse representations [18, 75] or approximate nearest neighbor (ANN) search in the case of dense representations [32], many existing works use pooling techniques to obtain a single vector representation for each query and document and then employ simple and light-weight scoring functions, such as inner product or cosine similarity. These also exist more expensive methods that do not use pooling and perform such light-weight scoring functions at the vector level and then aggregate them, such as the maximum inner product similarity used in ColBERT [33].

### On the Limitations of Linear Similarity Functions (e.g., Inner Product)

We believe the simple similarity functions used by existing bi-encoder models are not sufficient for modeling complex relationships between queries and documents. These functions inherently limit retrieval models to judge relevance in a way that can be represented by an inner product. Furthermore, it has been shown that the ability to compress and reconstruct information is correlated with the size, and thus complexity, of neural models [15]. This result indicates that using a relevance function as simple as an inner product likely reduces the amount of information that can be stored in a fixed representation size. These factors explain why state-of-the-art dense retrieval models continue to underperform cross-encoder models, in terms of retrieval quality [36]. In the following, we show the limitations of inner products (as a linear similarity function) by theoretically demonstrating the impossibility of inner products to produce perfect rankings for some queries, regardless of the method used to create the query and document embeddings.

Let C denote a corpus of N documents, each being represented by an h-dimensional vector. A perfect ranking of documents in C for a provided query is a ranking where all relevant documents are ranked above all non-relevant documents. According to Radon's Theorem [54], any set of h + 1 document vectors with h dimensions can be partitioned into two sets whose convex hulls intersect. An important application of Radon's Theorem is in calculating the Vapnik–Chervonenkis (VC) dimension [64] of h-dimensional vectors with respect to linear separability. For any h ≥ 2 vectors, the two subsets of a Radon partition cannot be linearly separated. In other words, for N > h + 1, there exists at least one group of documents that is not linearly separable from the rest. In the real world, since N > h + 1, there are indeed many such non-separable subsets. If any two of these subsets contain all the relevant documents for a query, then no linear similarity function can perfectly separate relevant from irrelevant documents. This includes inner product similarity and guarantees that, for some query, there will be an imperfect ranking.

To overcome these limitations with inner product similarity we use a multi-layer neural network with query-conditioned weights as our similarity measure. As neural networks are universal approximators [29], Hyperencoder's similarity function can express far more complex functions than those expressed by inner products. A related alternative approach with the same benefits takes the query and document representations, combines them (e.g., through concatenation or similar metrics), and feeds them to a neural network to serve as a similarity function (Figure 1 (second from the right)). However, this approach suffers from the following shortcomings: (1) query and document representations now need to be combined before scoring - adding latency proportional to the complexity of the method used to combine them; (2) having separate query and document representations increases the input dimension to the neural network further increasing latency; (3) for efficiency reasons, the query representation is often pooled or compressed before being input into the network which reduces the information the model receives. Hyperencoder addresses these shortcomings. Since the query is directly encoded as the neural network's weights no concatenation or other form of combining inputs is needed, the document representation can be directly input to the scoring network. This, in addition to the reduced network size from having only document representations as input, allows for a substantial latency improvement. Further, as Hyperencoder produces a query-specific neural network, every weight can be used to store query-related information without any need for compression or additional overhead. Lastly, we show in Section 3.7 that existing learned relevance methods can be exactly replicated by Hyperencoder with the additional flexibility of learning query-specific weights when desirable.

# 3.1 Hyperencoder Overview

An overview of our model is depicted in Figure 1 (right); it represents a new category of models that sit between a cross-encoder and a bi-encoder model. Like a bi-encoder model, our method computes the query and document representations separately, but unlike most existing retrieval methods, our method allows for more complicated matching signals like those present in cross-encoder models. Following existing methods, we have a query encoder and a document encoder. When a document is input into the document encoder, we obtain a representation similar to existing encoder models, namely a set of one or more vectors \(E_d \in \mathbb{R}^{n \times h}\) that represent the document’s content, where \(n\) is the number of vectors and \(h\) is the dimension of the vectors. Though we focus on vectors in this work, in theory, the representation can be anything a neural network can output.

Now comes our unique contribution that allows our method to consider more complex similarity signals. Given the query \(q\), the query encoder first produces a set of contextualized embeddings in a similar way to existing encoder models which we will call \(E_q \in \mathbb{R}^{n \times h}\), where \(n\) is the number of embeddings and \(h\) is the dimension of the embeddings. At this point while existing methods apply a simple pooling mechanism, our query encoder instead uses a hyper-head. The hyper-head takes \(E_q\) and produces a set of matrices and vectors that are then used as the weights and biases for a small neural network which we coin the q-net. The q-net is a query-dependent function for estimating relevance scores for each document embedding \(E_d^i\) that is input. The q-net is never created if, unlike existing neural scoring methods which use a shared set of weights for all queries. To find the relevance of a document, the document representation \(E_d\) is passed as input to the q-net which outputs the relevance score.

Hyperencoder is a generic framework which allows direct application of existing paradigms from neural retrieval and, more broadly, machine learning. For example, Hyperencoder could easily work with multiple vectors similar to existing multi-vector models, e.g., [33], or use training routines popularized in dense retrieval, e.g., [38, 52, 73, 76]. As an initial exploration, this paper focuses on showing the efficacy of Hyperencoder without additional complexity and thus uses a single vector document representation and no complex training recipes.

# 3.2 Query and Document Encoders

The Hyperencoder framework is generic and can be applied to any implementation of query and document encoders. In this work, we use pretrained transformer-based encoder models commonly used in the recent neural network literature. Specifically, we use a pre-trained BERT base model [16] for encoding queries and documents. Even though Hyperencoder can operate on all token representations produced for each document this work focuses on a single vector representation of documents, which is more efficient in terms of query latency, memory requirements, and disk usage. To do so, we can either use the contextualized embedding representing the [CLS] token or take the mean of all the contextualized embeddings for all the non-pad input tokens. Empirically, we found that using the [CLS] token performs better. Therefore, the document representation produced by the encoder is a single vector with 768 dimensions, i.e., the same as BERT’s output dimensionality. We refer to it as \(E_d \in \mathbb{R}^{n \times h}\), where \(n = 1\) is our setting.

Since Hyperencoder only uses the contextualized-query-token representations once to produce the q-net, it can skip pooling tokens without adding much cost. Therefore, we use all non-pad-token representations produced by the query encoder as the intermediate representation of the queries, denoted by \(E_q \in \mathbb{R}^{n \times h}\), where \(n\) is the number of tokens in the query \(q\) and \(h\) is the embedding dimensionality (\(h = 768\) in BERT).

# 3.3 The Hyperhead Layers

The method to transform \(E_q\) into the weights and biases for the q-net is performed by the hyperhead layers and is completely flexible. During our experimentation, we tried two mechanisms to do this transformation as well as many minor variants and found that both have stable training, which suggests the Hyperencoder framework is robust to the exact hyperhead layer implementation. Though we report both approaches, we settled on one for the final set of experiments in this paper which we will now describe.

For improved clarity, we focus only on the weight creation process as the biases are created in the exact same way. The contextualized query embeddings \(E_q \in \mathbb{R}^{n \times h}\) produced by the query encoder are independently transformed by l hyperhead layers, each of which corresponds to a layer in the q-net. Each hyperhead layer converts the embeddings \(E_q\) into key and value matrices:

\[
K_i^q = E_q \theta_{K_i} \quad \text{and} \quad V_i^q = E_q \theta_{V_i} \quad \forall i \in [0; l)
\]

where \(\theta_{K_i}, \theta_{V_i} \in \mathbb{R}^{h \times h}\) denote learnable parameters for computing key and value matrices. In the above equation, the embedding matrix \(E_q\) is concatenated with a column of all ones (i.e., \([E_q; 1]\)) to model both weight multiplication and bias addition.

Each key matrix \(K_i\) and value matrix \(V_i\) will be used for the creation of the weights in the \(i^{th}\) layer of the q-net. With the keys and values in hand, single-head scaled-dot-product attention [65] is performed using a query matrix \(Q_i \in \mathbb{R}^{h \times h}\) where \(h\) is the layer dimensionality in the \(i^{th}\) layer of q-net. In our case, all of the weights except the last layer are square matrices, making \(r = h\). Each \(Q_i\) is a set of learnable embeddings, similar to those used as input tokens for transformer models. Hence, the hidden layer representation \(H_i \in \mathbb{R}^{h \times h}\) is then computed as follows:

\[
H_i = \text{softmax} \left( \frac{Q_i K_i^T}{\sqrt{h}} \right) V_i
\]

A ReLU activation [1] is then applied to each \(H_i\) followed by layer normalization [4]. Next a point-wise feed-forward layer is applied to produce \(\hat{H}_i^q\):

\[
\hat{H}_i^q = \theta_{W_i} \text{L-Norm} (\text{ReLU}(H_i)) + \theta_{b_i}
\]

where L-Norm denotes layer normalization. Note that each weight in q-net has a unique \(\theta_{W_i}\) and \(\theta_{b_i}\). There are no learnable parameters in layer normalization.

The final operation to get the \(i^{th}\) weight \(W_i^q\) for q-net is:

\[
W_i^q = \hat{H}_i^q + \theta_{H_i}
\]

# 3.4 The q-net Network

Weights and biases produced by the hyperhead layers are not by themselves a neural network. They need a certain arrangement and additional components (e.g. non-linearity). This is where the Hypencoder’s q-net converter comes in. The converter knows the architecture of the q-net and given the weights and biases from the hyperhead layers, it produces a callable neural network object which takes as input the document representation \( E_d \).

It is worth highlighting that because the q-net’s architecture is not strictly tied to how the hyperhead layer produces the weights and biases, it is simple to modify the architecture of the q-net. All the hyperhead layers need to know is how many weights and biases are needed and what shape they should be.

In our experiments, we use a simple feed-forward architecture for the q-net. The output \( x^t_d \) from the input \( x^t_d \) at a given layer is given by:

\[
x^t_d = \text{L-Norm} \left( \text{ReLU} \left( W^t_q (x^t_d + b^t_q) \right) + x^t_d \right)
\]

where L-Norm represents a layer normalization without learnable parameters. The same equation is repeated for each layer, and a residual connection is applied before the final layer (i.e., layer \( l \)). The layer in Equation (6) is repeated \( l \) times. Finally, a relevance score is produced using a linear projection layer with an output dimensionality of 1.

# 3.5 Training

Training Hypencoder is no different from training a bi-encoder as it shares the same core components, i.e., a query encoder and document encoder. The only difference is instead of using an inner product to find the similarity the q-net is applied to the document representations. Thus, our contributions are solely the architecture and not a specific training technique. In this paper we employ a simple distillation training setup, for more details see Section 4.2.

# 3.6 Efficient Retrieval using Hypencoder

Being able to perform efficient retrieval is crucial for many real-world search scenarios where an exhaustive search is not feasible. For Hypencoder models, there is a clear parallel to dense models as both represent documents as dense vectors, but the differences between Hypencoder and dense models make it unclear whether the same efficient search techniques will work. For instance, it is clear that due to the linear nature of inner products, similar document vectors are likely to have similar inner products with a query vector; in the case of Hypencoder this assumption may not hold true as the non-linear nature of the Hypencoder scoring function could mean small differences in the input vector produce significant differences in the output score.

To study the extent to which Hypencoder’s retrieval can be approximated for efficient retrieval, we developed an approximate search technique based loosely on navigating small world graphs [35, 45]. In the index stage we construct a graph where documents are nodes connected to their neighbors by edges. We use \( l_2 \) distance between document embeddings similar to [62].

After constructing the document graph, approximate search is performed following Algorithm 1. In brief, a set of initial candidate documents \( C \) is selected at random, these candidates are scored with the q-net (line 5) and in lines 16-19 the best \( n \) candidates and their neighbors become the next candidates. In lines 12-15, the top scoring candidates are added to \( T \)—a set which stores the \( k \) best scoring documents so far. The algorithm terminates when one of three conditions is met: (1) the number of iterations equals maxIter; see line 4, (2) there are no more candidates; see line 4, or (3) no new documents are added to \( T \) at a given step; see line 8. We also consider an option without the final termination condition which we call without early stopping. As the number of operations is dependent on the number of initial candidates \( |C| \), the running time is not tied to the number of documents, resulting in a run time complexity of \( O(|C| + n \times \text{candidates} \times \text{maxIter}) \).

With this algorithm, we found that Hypencoder is able to significantly increase retrieval speed without a large loss in quality. See the results in Section 4.4.4.

## Algorithm 1 Hypencoder Efficient Search

- **Input**: q-net \( q \), NN to return \( k \), initial candidates \( C \), candidates to explore every iteration \( n \), maxIter, neighbors
- **Output**: \( T \) = top \( k \) documents

```plaintext
1:  \( v = C \)  // Set of visited elements
2:  \( T = (-\infty) \)  // Stores top \( k \) nearest neighbors to \( q \) at any given time
3:  \( i = 0 \)  // Current iteration
4:  while \( |C| > 0 \) and \( i < \text{maxIter} \) do
5:      \( c = \) find top \( n \) candidates values in \( C \) using \( q \)
6:      \( f = \) get lowest scoring element from \( T \)
7:      if max\_score \( c < f \) then
8:          break  // All candidates are worse than \( T \) so stop now
9:      \( C = \{\} \)  // Reset \( C \)
10:     for each \( e \in c \) do
11:         \( f = \) get lowest scoring element from \( T \)
12:         if \( q(e) > f \) or \( |T| < k \) then
13:             \( T = T \cup e \)
14:         if \( |T| > k \) then
15:             \( T = T \setminus \{f\} \)
16:         for each \( n \in \text{NEIGHBORS}(e) \) do
17:             if \( n \notin e \) then
18:                 \( C = C \cup n \)
19:                 \( v = v \cup n \)
20:     \( i = i + 1 \)
21: return \( T \)
```

# 3.7 Comparison to Existing Neural IR Methods

We argue that Hypencoder can exactly reproduce existing neural ranking models. Let us start by formalizing the main components of existing neural methods: (1) a query representation \( E_q \)

# 4 EXPERIMENTS

## 4.1 Datasets

The dataset used for training our models is the training split of the MSMARCO passage retrieval dataset [4] that contains 8.8M passages and has 533K training queries with at least one corresponding relevant passage. The queries in the MSMARCO training set are short natural language questions asked by users of the Bing search engine.

To create the training pairs, we first retrieved 500 passages for every query using an early iteration of Hypencoder. From these, we sampled 200 passages — the top 100 passages and another 100 randomly sampled from the remaining 400 passages. These query-passage pairs were then labeled using the MiniLM cross-encoder from the Sentence Transformers Library [55].

### 4.1.2 Validation Dataset

For validating and parameter tuning, we use the TREC Deep Learning (DL) 2021 [11] and 2022 passage task [11, 12], as the passage collection for TREC DL '21 and '22 is large and we wanted validation to be fast we created a subset with only passages in the QREL files.

### 4.1.3 Evaluation Datasets

Our evaluation explores retrieval performance in three different areas: in-domain performance, out-of-domain performance, and performance on hard retrieval tasks.

For in-domain performance, we use the MSMARCO Dev set [49], TREC Deep Learning 2019 [13], and TREC Deep Learning 2020 [10]. The MSMARCO Dev set contains around 7k queries with shallow labels, the majority of queries only have a single passage labeled as relevant. This collection uses queries from the same distribution as the training queries making it a clear test of the in-domain performance. On this dataset we report the standard evaluation metrics: MRR and Recall@1000. The TREC Deep Learning 2019 and 2020 datasets have a similar query distribution to MSMARCO Dev but feature far fewer queries, i.e., 97 queries combined. The lower number of queries is compensated by far deeper annotations with every query having several annotated passages.

To assess out-of-domain performance, we evaluate on question answering tasks on different domains, specifically, the TREC COVID [56] and NFCorpus datasets [7] for the biomedical domain and FiQA [44] for the financial domain. We also evaluate on DBPedia [23] as an entity retrieval dataset and on Touché [6] as an argument retrieval dataset. We use the BEIR [63] versions of these datasets from the ir_datasets library. 

To explore the full capabilities of Hypencoder we want to evaluate how it performs on retrieval tasks that are more challenging than standard question-passage retrieval tasks. To some extent hardness is subjective, but we tried to define a clear process to define difficulty: (1) current neural retrieval models should struggle on the task, (2) term matching models like BM25 should also struggle on the task, (3) the queries are longer or otherwise more complicated than standard web queries. An additional requirement we had was that for tasks that were significantly different from the MSMARCO training data we wanted adequate training data to finetune the models before evaluation. We believe this is reasonable as we are not investigating the models' zero-shot performance but the inherent limits of the model.

The first dataset we select was the TREC Tip-of-the-Tongue (TOT) 2023 [3] that contains queries with user intent to find many aspects of the item they were looking for but not the name of the item. Thus TOT queries tend to be verbose and can include many facets. The TREC 2023 dataset specifically looks at TOT queries for movies with the corresponding movie's Wikipedia page as the golden passage. We use the development set as the test relevance labels are not public yet. There are 150 queries. Each query has a single relevant passage. For training we use the data from Bhargav et al. [5] which is around 15k TOT queries from Reddit for the book and movie domain.

The second dataset is FollowIR [68] for instruction following retrieval. This dataset is built on top of three existing TREC datasets: TREC Robust '04 [67], TREC News '21 [60], and TREC Core '17 [2]; it uses the fact that these datasets include instructions to the annotators which can act as a complex instruction. To test how well a retrieval system follows the instruction the creators of FollowIR modify the instruction to be more specific and re-annotate the known relevant documents. As training data we use MSMARCO with Instructions, a recent modification of MSMARCO which adds instructions to the queries as well as new positive passages and hard negative passages which consider the instruction [70].

The final dataset is a subset of TREC DL-HARD [43]. The full dataset uses some of the queries from TREC DL 2019 and 2020 as well as some queries that were considered for DL 2019 and 2020 but were not included in the final query collection. TREC DL-HARD is built specifically with the hardest queries from the TREC DL pool. The authors do so by using a commercial search engine to find queries that are not easily answered. The standard TREC DL-HARD dataset has 50 queries half of which appear in TREC DL

# Hyperencoder: Hypernetworks for Information Retrieval

**Figure 2**: Relationship between the three main parameters of our efficient search: the size of the initial set of candidates \(\hat{C}\), the number of neighbors to explore \(ncandidates\), and the number of iterations \(maxIter\) and both effectiveness in terms of TREC DL '19 nDCG@10 and efficiency in terms of Query Latency.

## 4.2 Experimental Setup

### 4.2.1 Training Details

All the Hyperencoders use BERT [16] base uncased as the base model. We use PyTorch and Huggingface for model implementation and training. All of our q-nets use a input dimension of 768 and hidden dimension of 768. Unless otherwise stated, we use 6 linear layers in the model not including the final output projection layer.

We use a training batch size of 64 per device and 128 in total. A single example in the batch is a query, positive document, and 8 additional documents ranking in relevance. The positive document is the top ranked document by our teacher model. The other documents are sampled randomly from the passages associated with the query. For more details about the dataset see Section 4.1.1. Passages were truncated to 196 tokens and queries to 32 tokens.

Our primary loss function is Margin MSE [26]. When computing the loss, we construct (query, positive document, negative document) triplets where all of the negatives for a query form their own triplet. The loss is found by averaging the individual loss of all triplets. In addition to Margin MSE, we use an in-batch cross entropy loss where the (query, positive document) is assumed to be a true positive and all the other queries' positive documents are assumed to be negatives. We do not consider the additional "hard" negatives from the query in the cross entropy loss as many of these documents are relevant to the query. We use AdamW as our optimizer with default settings and a learning rate of 2e-5 with a constant scheduler after a warm up of 6k steps. Our training hardware consists of two A100 GPUs with 80GB memory. Training took around 6 days.

To select the best model we evaluate each model on the validation set every 50k steps and pick the model with the best R Precision within the first 800k steps. We selected R Precision due to the fact that it balances both recall and precision in a single metric and does not require a specified cutoff depth.

When training for the harder tasks we use AdamW with the learning rate 8e-6 with a linear scheduler and a warm-up ratio of 1e-1. For TOT training we train for 25 epochs or 3.3k steps. For FollowIR training we train 1 epoch for around 10k steps. We use a batch-size of 96 and cross entropy loss. Each example in the batch includes a query, positive document, and hard negative document. We use a maximum document and query length of 512 tokens.

## 4.3 Baselines

For comparison with Hyperencoder, we include several baseline models and models which we include for reference which are not directly comparable. Our main baselines which we evaluate on all datasets are TAS-B [28], CIL-DRD [76], BM25, and our own bi-encoder baseline which we call BE-Base. We train BE-Base exactly the same as Hyperencoder except we use separate encoders and use a linear LR scheduler. We select our main dense baselines TAS-B and CIL-DRD as they are both strong bi-encoder models which leverage knowledge-distillation training and which use the same document embedding dimension of 768. For in-domain results we include an additional set of dense models: ANCE [73], TCT-ColBERTv2 [37], and MarginMSE [26]. We only include these models in the in-domain results to save space in other sections and because TAS-B and CIL-DRD outperform the other baselines. For reference we also include: the late-interaction model ColBERTv2 [5]; the neural sparse model SPLADE++ [3] [17]; RepL4MA a 7B parameter bi-encoder model [41]; DRAGON a bi-encoder trained with 5 teacher models and 20M synthetic queries; MonoBERT a re-ranking model reranking the top 1k BM25 retrievals [50]; and the reranking model cross-SimLM reranking the top 200 passages from bi-SimLM [68]. Reference results were taken from the RepL4MA [41] and DRAGON [38] papers.

## 4.4 Results and Discussion

### 4.4.1 In-Domain Results

Our in-domain results are presented in Table 1; they demonstrate that compared with baselines and even the reference models Hyperencoder has very strong performance. Hyperencoder is significantly better than each baseline in nDCG@10 on the combined TREC '19, '20 and '21 and statistically better than all but CIL-DRD on MSMARCO Dev R@10. The most direct comparison, BE-Base, has far lower nDCG@10, R@R, and R@10 values indicating the Hyperencoder is able to bring a large boost in precision based metrics over dense retrieval. In terms of recall Hyperencoder is either as good or better than all the baselines though the gap is not as large as for precision based metrics.

# Table 1: Comparison on in-domain evaluation datasets

The symbols next to each baseline indicate significance values with \( p < 0.05 \). Note, that † is a group of baselines.

| Model                        | TREC-DL '19 & '20 | MSAMRCO Dev |
|------------------------------|-------------------|-------------|
|                              | nDCG@10           | RR@1000     | nDCG@10 | RR@1000 |
| **Single Vector Dense Retrieval Models & BM25 (Baseline)** | | | | |
| BM25 †                       | 0.491             | 0.679       | 0.735   | 0.184   | 0.853   |
| ANCE †                       | 0.686             | 0.611       | 0.767   | 0.330   | 0.958   |
| TCT-ColBERT †                | 0.669             | 0.820       | 0.860   | 0.335   | 0.964   |
| Margin MSE †                 | 0.700             | 0.853       | 0.782   | 0.352   | 0.955   |
| TAS-B †                      | 0.709             | 0.863       | 0.851   | 0.344   | 0.978   |
| CL-DRB †                     | 0.701             | 0.841       | 0.838   | 0.382   | 0.981   |
| BE-Base †                    | 0.713             | 0.855       | 0.826   | 0.359   | 0.968   |
| **Hypencoder**               | 0.736\*           | 0.885\*     | 0.871\* | 0.368\* | 0.984\* |
| **Other Retrieval Models (Reference Models)** | | | | |
| ColBERTv2                    | 0.749             | -           | -       | 0.397   | 0.984   |
| SPLADE++                     | 0.773             | -           | -       | 0.368   | 0.979   |
| RepL4LM-a                    | 0.731             | -           | -       | 0.412   | 0.944   |
| DRAGON                       | 0.734             | -           | -       | 0.393   | 0.985   |
| MonoBERT                     | 0.772             | -           | -       | 0.372   | 0.953   |
| cross-BLM                    | 0.735             | -           | -       | 0.437   | 0.927   |

# Table 2: Out-of-domain results in nDCG@10

We only compare significance with B-base. Significance results with \( p < 0.05 \) are shown with \# and \( p < 0.1 \) are shown with \@.

| Rep type | Baselines | Ours |
|----------|-----------|------|
|          | BM25      | TAS-B | CL-DRB | BE-Base | Hypencoder |
| Q & A    |           |       |        |         |            |
| TREC-Covid | 0.566   | 0.481 | 0.584  | 0.651   | 0.685\#    |
| FIQA     | 0.256     | 0.300 | 0.308  | 0.309   | 0.314      |
| NFCorpus | 0.325     | 0.319 | 0.315  | 0.327   | 0.324      |
| Misc.    |           |       |        |         |            |
| DBPedia  | 0.313     | 0.364 | 0.381  | 0.405   | 0.419\#    |
| ToucheV2 | 0.367     | 0.162 | 0.203  | 0.240   | 0.258\@    |

Impressively Hypencoder is able to surpass DRAGON on nDCG@10 on the combined TREC DL '19 and '20 query set. Hypencoder uses the same base model and is a bi-encoder, it uses 32 A100s to train, with no training queries, and a complex 5 teacher curriculum learning distillation training technique. In other words, DRAGON is likely close to the limit of existing BERT-based bi-encoders and still Hypencoder is able to outperform it with a different domain setup and far less training compute.

Hypencoder also beats both rerankers MonoBERT and SimLMv2, demonstrating that reranking cannot make up for a weak retriever's performance. Continuing in TREC-DL '19 and '20 we find that Hypencoder even surpassed RepL4LM-a which is more than 60x larger and which also uses a significantly larger document embedding dimension of 4096. In fact the only model beating Hypencoder in nDCG@10 is ColBERTv2 which uses an embedding for every token in the document compared to Hypencoder's fixed 768 dimension token. MSAMRCO Dev results are also good with Hypencoder outperforming all the baselines and outperforming a few of the reference models such as SPLADE++ and MonoBERT.

Overall Hypencoder's in-domain results are exceptionally strong given the simple training routine used, small encoder model size, and document representation size. To the best of our knowledge, Hypencoder sets a new record for combined TREC-DL '19 and '20 nDCG@10 with a 768 dimension dense document vector.

## 4.4.2 Out-of-Domain Results

Table 2 shows our results on the select out-of-domain datasets, we only include our main baseline models and BM25 due to space limitations. The general trend is that Hypencoder has strong out-of-domain performance in question answering tasks (Q&A) and entity retrieval tasks. This indicates that despite Hypencoder's more complex similarity function it is still able to generalize well in a zero-shot manner to new tasks.

## 4.4.3 Results on Harder Retrieval Tasks

The results on the harder retrieval tasks are in Table 3, like in the out-of-domain section we only consider the main baseline models and BM25. We can see that in the harder tasks Hypencoder remains dominant over the baseline models with higher retrieval metrics in all but one column. Additionally, the relative improvement compared to the in-domain results are higher (on all metrics that Hypencoder is the best for) suggesting that on harder tasks the added complexity that can be captured by Hypencoder’s similarity function is especially important. Additionally the high performance on TREC tip-of-the-tongue (TOT) and FollowIR indicate that Hypencoder adapts well to different domains through domain-specific fine-tuning.

On the overall subset of TREC-DL HARD we see that Hypencoder has stronger precision metrics than the baselines by a large margin. This suggests that Hypencoder is especially dominant on harder tasks which, in part, explains its higher performance on TREC DL '19 and '20. Though on in-domain dataset Hypencoder does better on the same on recall metrics, on TREC DL-HARD BE-Base has higher recall than Hypencoder. This may be because the relevance function that the q-pt applies is not smooth, which has the benefit of being more discerning and likely accounts for some of the precision gains. However, if the q-pt makes a mistake the non-smooth scoring could result in a much harsher score than the linear inner product is capable of producing.

Moving to TREC tip-of-my-tongue (TOT) we see that Hypencoder continues to perform well. Tip-of-the-tongue is a complex retrieval task with long queries and passages and multiple aspects, the fact Hypencoder outperforms the baselines by a large margin validates the need for a more complex relevance function.

Finally we have FollowIR which has three subsets – on all three Hypencoder has the best performance on the retrieval evaluation metrics of choice, in many cases by a sizable amount. Beside the retrieval evaluation metrics we also include p-MRR which is a metric released in the FollowIR [69] paper. The metric measures the change in document ranks before and after an instruction is modified to see how well the model responses to the additional requirements. A p-MRR of 0 indicates no change in document rank based on the instruction change and a p-MRR of +100 indicates the documents were perfectly changed based on the instruction while -100 indicates the opposite. For additional details we refer readers to the original FollowIR paper [69]. As p-MRR is relative to each model’s performance before the instructions are modified it is not indicative of stand-alone retrieval performance. With that said, Hypencoder is the only model to achieve a positive p-MRR.

# Table 3: Evaluation metrics for the harder set of tasks

| Model        | TREC DL-HARD | TREC TOT DEV | FollowIR Robust '04 | FollowIR News '21 | FollowIR Core '17 |
|--------------|--------------|--------------|---------------------|-------------------|-------------------|
|              | nDCG@10      | RR           | nDCG@10             | AP                | p-MRR             |
| BM25 †       | 0.646        | 0.813        | 0.646               | 0.086             | 0.131             | 0.121             |
| TAS-B ♠      | 0.574        | 0.789        | 0.777               | 0.097             | 0.089             | 0.162             | 0.203             | -5.4  |
| CL-DRD ♠     | 0.573        | 0.790        | 0.719               | 0.088             | 0.062             | 0.151             | 0.206             | -7.2  |
| BE-Base ♠    | 0.607        | 0.864        | 0.805               | 0.121             | 0.110             | 0.179             | 0.207             | -3.7  |
| **Hyperencoder** | **0.630†** | **0.887♠** | **0.798†** | **0.134†♠** | **0.125♠** | **0.182†♠** | **0.212♠** | **-3.5** | 0.272 | 2.0 | 0.193 | -11.8 |

Significance is shown at p < 0.1. For FollowIR we do not perform significance tests on BM25.

----

# Table 4: Average query latency and nDCG@10 on TREC DL '19 and '20 with efficient search

Efficient 1 uses parameters (Ĉ = 10000, nCandidates = 64, maxIter = 16), Efficient 2 uses parameters (Ĉ = 10000, nCandidates = 328, maxIter = 20). All model inference was performed on an NVIDIA L40S with BF16 precision.

| Search Type | Query Lat. (ms) | DL '19 | DL '20 |
|-------------|-----------------|--------|--------|
| Exhaustive  | 1766            | 0.82   | 0.731  |
| Efficient 1 | 596.1           | 0.722  | 0.730  |
| Efficient 2 | 231.1           | 0.722  | 0.731  |

----

# Figure 3: Average nDCG@10 on TREC DL '19 and '20 versus number of layers in the q-net

| Number of q-net layers | nDCG@10 |
|------------------------|---------|
| 2                      | 0.74    |
| 4                      | 0.73    |
| 6                      | 0.72    |
| 8                      | 0.72    |

----

# 5 CONCLUSION

We propose a new class of retrieval model, the Hyperencoder which overcomes the limitations of inner product based similarity functions that we prove to exist. Our model achieves a new state-of-the-art on TREC DL '19 and '20 for BERT sized encoder models with a single dense document vector and shows even stronger relative improvement on harder retrieval tasks such as tip-of-the-tongue queries. Further we demonstrate that learned relevance models can be applied to large-scale search corpora in an efficient way with our proposed approximate search algorithm. As Hyperencoder is a flexible framework there is much interesting future work to explore, such as multi-vector document representations and corpus pretraining to name a few.

# ACKNOWLEDGMENTS

This work was supported in part by the Center for Intelligent Information Retrieval, in part by the NSF Graduate Research Fellowship Program (GRFP) Award #1938059, and in part by the Office of Naval Research contract number N000142112612. Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect those of the sponsor.

# REFERENCES

[1] Aiden Fred Argus. 2018. Deep Learning using Rectified Linear Units (ReLU). CoRR abs/1803.08375 (2018). arXiv:1803.08375. http://arxiv.org/abs/1803.08375

[2] James Allan, James P. Kramer, R. Manmatha, Lori L. Christiansen Gyen, and Ellen M. Voorhees. 2017. The 2017 Common Core Track Overview. In Text Retrieval Conference. https://sigir.org/documents/overview2017.pdf

[3] Bhaskar Mitra, Santu Dhar, Hugo Zaragoza, Fernando Diaz, Evangelos Kanoulas, and Nazli Goharian. 2022. Overview of the TREC 2022 Deep Learning Track. In The Thirtieth Text REtrieval Conference Proceedings (TREC 2023), Gaithersburg, MD, USA, November 14-17, 2022 (NIST Special Publication, Vol. 126). Ian Soboroff, James Allan, and Angela Ellis (Eds.). National Institute of Standards and Technology (NIST). https://trec.nist.gov/pubs/trec31/papers/Overview.DL.pdf

[4] Lee Jimmy He, Jamie Ryan Kross, and Geoffrey D. Hinton. 2016. Layer Normalization. CoRR abs/1607.06450 (2016). arXiv:1607.06450. http://arxiv.org/abs/1607.06450

[5] Samarth Bhargav, Georgios Sidiropoulos, and Evangelos Kanoulas. 2022. "It’s on the Tip of my Tongue": A New Dataset for Known-Item Search Evaluation. In The Fifteenth ACM International Conference on Web Search and Data Mining, Virtual Event, France, 21-25, February 21 - 25, 2022, É. K. Schedl, Claudia Hauff, Ilya Leonov, J. M. Jose, and Yi Liu (Eds.). ACM, 123-126. https://doi.org/10.1145/3488560.3498224

[6] Alexander Bondarenko, Malik Firoz, Meriem Beloucif, Lukas Gienapp, Yamen Ajjour, Christopher Michael Klaus, Hinrich Schütze, and Benno Stein. 2022. Webis-CLS-10: A Large-Scale Multilingual Dataset for Cross-Language Summarization. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Dublin, Ireland, May 22-27, 2022. Association for Computational Linguistics, 2208-2229. https://doi.org/10.18653/v1/2022.acl-long.158

[7] Yamen Ajjour, Alexander Bondarenko, Malik Firoz, Meriem Beloucif, Lukas Gienapp, and Benno Stein. 2022. Webis-CLS-10: A Large-Scale Multilingual Dataset for Cross-Language Summarization. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Dublin, Ireland, May 22-27, 2022. Association for Computational Linguistics, 2208-2229. https://doi.org/10.18653/v1/2022.acl-long.158

[8] Yamen Ajjour, Alexander Bondarenko, Malik Firoz, Meriem Beloucif, Lukas Gienapp, and Benno Stein. 2022. Webis-CLS-10: A Large-Scale Multilingual Dataset for Cross-Language Summarization. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Dublin, Ireland, May 22-27, 2022. Association for Computational Linguistics, 2208-2229. https://doi.org/10.18653/v1/2022.acl-long.158

[9] Yamen Ajjour, Alexander Bondarenko, Malik Firoz, Meriem Beloucif, Lukas Gienapp, and Benno Stein. 2022. Webis-CLS-10: A Large-Scale Multilingual Dataset for Cross-Language Summarization. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Dublin, Ireland, May 22-27, 2022. Association for Computational Linguistics, 2208-2229. https://doi.org/10.18653/v1/2022.acl-long.158

[10] Yamen Ajjour, Alexander Bondarenko, Malik Firoz, Meriem Beloucif, Lukas Gienapp, and Benno Stein. 2022. Webis-CLS-10: A Large-Scale Multilingual Dataset for Cross-Language Summarization. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Dublin, Ireland, May 22-27, 2022. Association for Computational Linguistics, 2208-2229. https://doi.org/10.18653/v1/2022.acl-long.158

[11] Yamen Ajjour, Alexander Bondarenko, Malik Firoz, Meriem Beloucif, Lukas Gienapp, and Benno Stein. 2022. Webis-CLS-10: A Large-Scale Multilingual Dataset for Cross-Language Summarization. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Dublin, Ireland, May 22-27, 2022. Association for Computational Linguistics, 2208-2229. https://doi.org/10.18653/v1/2022.acl-long.158

[12] Yamen Ajjour, Alexander Bondarenko, Malik Firoz, Meriem Beloucif, Lukas Gienapp, and Benno Stein. 2022. Webis-CLS-10: A Large-Scale Multilingual Dataset for Cross-Language Summarization. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Dublin, Ireland, May 22-27, 2022. Association for Computational Linguistics, 2208-2229. https://doi.org/10.18653/v1/2022.acl-long.158

[13] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, and Jimmy Lin. 2022. Overview of the TREC 2022 Deep Learning Track. In Proceedings of the Thirtieth Text REtrieval Conference, TREC 2022, Virtual Event (Gaithersburg, Maryland, USA), November 14-17, 2022 (NIST Special Publication, Vol. 126). Ian Soboroff, Ellen M. Voorhees, and Angela Ellis (Eds.). National Institute of Standards and Technology (NIST). https://trec.nist.gov/pubs/trec31/papers/Overview.DL.pdf

[14] Mostafa Dehghani, Hamed Zamani, Alireza Seyyedi, Jay Kamps, and W. Bruce Croft. 2017. Neural Ranking Models with Weak Supervision. In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR), Tokyo, Japan. SIGIR '17. Association for Computing Machinery, New York, NY, USA, 65–74. https://doi.org/10.1145/3077136.3080836

[15] Giorgio Deléger, Anjan Rous, Paul-Ambroise Duquenne, Elliot Cat, Tim Ginevra, Christopher Manteau, and Gran-Moya. Livia K. Weinling, Matthew A. Christensen, Laurent Orsini, Marcus Tuttle, and Joel Veness. 2022. Language Modeling is Compression. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net. https://openreview.net/forum?id=3g3v3v3v3

[16] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019. (Long and Short Papers). Association for Computational Linguistics, 4171–4186. https://doi.org/10.18653/v1/N19-1423

[17] Thibault Formal, Carlos Lassance, Hugues Bouchard, and Stephane Clinchant. 2022. From Distillation to Fine Tuning: Improving Neural Ranker Models for Relevance Estimation. In SIGIR '22: The 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, Madrid, Spain, July 11-15, 2022. Association for Computing Machinery, New York, NY, USA, 2335–2339. https://doi.org/10.1145/3477495.3531837

[18] Thibault Formal, Benjamin Prowowski, and Stephane Clinchant. 2021. SPALDE: Sparse Attention for Language Models. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Punta Cana, Dominican Republic, November 7-11, 2021. Association for Computational Linguistics, 1234–1245. https://doi.org/10.18653/v1/2021.emnlp-main.98

[19] Luyu Gao and Jamie Callan. 2022. Unsupervised Corpus Aware Language Model Pre-training for Dense Passage Retrieval. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Dublin, Ireland, May 22-27, 2022. Association for Computational Linguistics, 2843–2853. https://doi.org/10.18653/v1/2022.acl-long.203

[20] Luyu Gao, Xinyu Fan, Liang Pang, Lian Yang, Qingyao Ai, Hamed Zamani, Cheng Luo, W. Bruce Croft, and Jaejoong Lee. 2022. A Deep Look into Neural Ranking Models for Information Retrieval. Information Processing & Management 59, 2 (2022), 100627. https://doi.org/10.1016/j.ipm.2021.102627

[21] Luyu Gao, Xinyu Fan, Liang Pang, Lian Yang, Qingyao Ai, Hamed Zamani, Cheng Luo, W. Bruce Croft, and Jaejoong Lee. 2022. A Deep Look into Neural Ranking Models for Information Retrieval. Information Processing & Management 59, 2 (2022), 100627. https://doi.org/10.1016/j.ipm.2021.102627

[22] Xiangnan He and Tat-Seng Chua. 2017. Neural Factorization Machines for Sparse Predictive Analytics. In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval, Shinjuku, Tokyo, Japan, August 7-11, 2017. Worio Kando, Tetsuya Sakai, Hideo Joho, Hang Li, Arjen P. de Vries, and Ryen W. White (Eds.). ACM, 355–364. https://doi.org/10.1145/3077136.3080775

[23] Xiangnan He, Lizi Lian, Hanwang Zhang, Lichang Xie, Niu Xu, Hua and Tat-Seng Chua. 2017. Neural Collaborative Filtering. In Proceedings of the 26th International Conference on World Wide Web, WWW 2017, Perth, Australia, April 3-7, 2017. Rick Barrett, Rick Cummings, Eugene Agichtein, and Evgeniy Gabrilovich (Eds.). ACM, 173–182. https://doi.org/10.1145/3038912.3052569

[24] Sebastian Hofstätter, Sophia Althammer, Michael Schröder, Mete Sertkan, and Allan Hanbury. 2020. Improving Efficient Neural Rankers with Cross-Architecture Knowledge Distillation. CoRR abs/2012.00666 (2020). arXiv:2012.00666. https://arxiv.org/abs/2012.00666

[25] Sebastian Hofstätter, Katharina Schöpf, Sophia Althammer, Mete Sertkan, and Allan Hanbury. 2022. Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling. In SIGIR '22: The 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, Madrid, Spain, July 11-15, 2022. Association for Computing Machinery, New York, NY, USA, 113–122. https://doi.org/10.1145/3477495.3531837

NO_CONTENT_HERE

NO_CONTENT_HERE