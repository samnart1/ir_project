#!/usr/bin/env python3
"""
Generate a comprehensive ArXiv-style corpus of ML/AI papers.

This creates a realistic dataset based on real paper metadata
for use in the IR project evaluation.
"""

import json
import random
import os

# Real ML/AI papers with titles and abstracts
PAPERS_DATA = [
    # Transformers & Attention
    {"title": "Attention Is All You Need", "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.", "categories": ["cs.CL", "cs.LG"], "year": 2017},
    {"title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks.", "categories": ["cs.CL"], "year": 2019},
    {"title": "GPT-2: Language Models are Unsupervised Multitask Learners", "abstract": "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on task-specific datasets. We demonstrate that language models begin to learn these tasks without any explicit supervision when trained on a new dataset of millions of webpages called WebText. Our largest model, GPT-2, is a 1.5B parameter Transformer that achieves state of the art results on 7 out of 8 tested language modeling datasets.", "categories": ["cs.CL", "cs.LG"], "year": 2019},
    {"title": "Language Models are Few-Shot Learners", "abstract": "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance.", "categories": ["cs.CL"], "year": 2020},
    {"title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach", "abstract": "Language model pretraining has led to significant performance gains but careful comparison between different approaches is challenging. Training is computationally expensive, often done on private datasets of different sizes, and hyperparameter choices have significant impact on the final results. We present a replication study of BERT pretraining that carefully measures the impact of many key hyperparameters and training data size.", "categories": ["cs.CL"], "year": 2019},
    {"title": "XLNet: Generalized Autoregressive Pretraining for Language Understanding", "abstract": "With the capability of modeling bidirectional contexts, denoising autoencoding based pretraining like BERT achieves better performance than pretraining approaches based on autoregressive language modeling. However, relying on corrupting the input with masks, BERT neglects dependency between the masked positions and suffers from a pretrain-finetune discrepancy. In light of these pros and cons, we propose XLNet, a generalized autoregressive pretraining method.", "categories": ["cs.CL", "cs.LG"], "year": 2019},
    {"title": "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations", "abstract": "Increasing model size when pretraining natural language representations often results in improved performance on downstream tasks. However, at some point further model increases become harder due to GPU/TPU memory limitations and longer training times. To address these problems, we present two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT.", "categories": ["cs.CL"], "year": 2020},
    {"title": "DistilBERT: A Distilled Version of BERT", "abstract": "As Transfer Learning from large-scale pre-trained models becomes more prevalent in Natural Language Processing, operating these large models in on-the-edge and/or under constrained computational training or inference budgets remains challenging. In this work, we propose a method to pre-train a smaller general-purpose language representation model, called DistilBERT, which can then be fine-tuned with good performances on a wide range of tasks.", "categories": ["cs.CL"], "year": 2019},
    {"title": "T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", "abstract": "Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts all text-based language problems into a text-to-text format.", "categories": ["cs.CL", "cs.LG"], "year": 2020},
    {"title": "Reformer: The Efficient Transformer", "abstract": "Large Transformer models routinely achieve state-of-the-art results on a number of tasks but training these models can be prohibitively costly, especially on long sequences. We introduce two techniques to improve the efficiency of Transformers. For one, we replace dot-product attention by one that uses locality-sensitive hashing, changing its complexity from O(L^2) to O(L log L), where L is the length of the sequence.", "categories": ["cs.LG", "cs.CL"], "year": 2020},
    {"title": "Longformer: The Long-Document Transformer", "abstract": "Transformer-based models are unable to process long sequences due to their self-attention operation, which scales quadratically with the sequence length. To address this limitation, we introduce the Longformer with an attention mechanism that scales linearly with sequence length, making it easy to process documents of thousands of tokens or longer.", "categories": ["cs.CL"], "year": 2020},
    {"title": "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators", "abstract": "Masked language modeling (MLM) pre-training methods such as BERT corrupt the input by replacing some tokens with a special token and then train a model to reconstruct the original tokens. While they produce good results when transferred to downstream NLP tasks, they generally require large amounts of compute to be effective. As an alternative, we propose a more sample-efficient pre-training task called replaced token detection.", "categories": ["cs.CL", "cs.LG"], "year": 2020},
    
    # Computer Vision
    {"title": "Deep Residual Learning for Image Recognition", "abstract": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth.", "categories": ["cs.CV"], "year": 2016},
    {"title": "ImageNet Classification with Deep Convolutional Neural Networks", "abstract": "We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 37.5% and 17.0% which is considerably better than the previous state-of-the-art. The neural network, which has 60 million parameters and 650,000 neurons, consists of five convolutional layers.", "categories": ["cs.CV", "cs.NE"], "year": 2012},
    {"title": "Very Deep Convolutional Networks for Large-Scale Image Recognition", "abstract": "In this work we investigate the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting. Our main contribution is a thorough evaluation of networks of increasing depth using an architecture with very small 3x3 convolution filters, which shows that a significant improvement on the prior-art configurations can be achieved by pushing the depth to 16-19 weight layers.", "categories": ["cs.CV"], "year": 2015},
    {"title": "Going Deeper with Convolutions", "abstract": "We propose a deep convolutional neural network architecture codenamed Inception that achieves the new state of the art for classification and detection in the ImageNet Large-Scale Visual Recognition Challenge 2014. The main hallmark of this architecture is the improved utilization of the computing resources inside the network. By a carefully crafted design, we increased the depth and width of the network while keeping the computational budget constant.", "categories": ["cs.CV"], "year": 2015},
    {"title": "Densely Connected Convolutional Networks", "abstract": "Recent work has shown that convolutional networks can be substantially deeper, more accurate, and efficient to train if they contain shorter connections between layers close to the input and those close to the output. In this paper, we embrace this observation and introduce the Dense Convolutional Network (DenseNet), which connects each layer to every other layer in a feed-forward fashion.", "categories": ["cs.CV", "cs.LG"], "year": 2017},
    {"title": "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", "abstract": "Convolutional Neural Networks are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance. Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient.", "categories": ["cs.CV", "cs.LG"], "year": 2019},
    {"title": "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications", "abstract": "We present a class of efficient models called MobileNets for mobile and embedded vision applications. MobileNets are based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural networks. We introduce two simple global hyperparameters that efficiently trade off between latency and accuracy.", "categories": ["cs.CV"], "year": 2017},
    {"title": "ShuffleNet: An Extremely Computation-Efficient CNN for Mobile Devices", "abstract": "We introduce an extremely computation-efficient CNN architecture named ShuffleNet, which is designed specially for mobile devices with very limited computing power. The new architecture utilizes two new operations, pointwise group convolution and channel shuffle, to greatly reduce computation cost while maintaining accuracy.", "categories": ["cs.CV"], "year": 2018},
    {"title": "Squeeze-and-Excitation Networks", "abstract": "The central building block of convolutional neural networks is the convolution operator, which enables networks to construct informative features by fusing both spatial and channel-wise information within local receptive fields at each layer. In this paper, we focus instead on the channel relationship and propose a novel architectural unit, which we term the Squeeze-and-Excitation block, that adaptively recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.", "categories": ["cs.CV"], "year": 2018},
    {"title": "YOLO: You Only Look Once Real-Time Object Detection", "abstract": "We present YOLO, a new approach to object detection. Prior work on object detection repurposes classifiers to perform detection. Instead, we frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance.", "categories": ["cs.CV"], "year": 2016},
    {"title": "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks", "abstract": "State-of-the-art object detection networks depend on region proposal algorithms to hypothesize object locations. Advances like SPPnet and Fast R-CNN have reduced the running time of these detection networks, exposing region proposal computation as a bottleneck. In this work, we introduce a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals.", "categories": ["cs.CV"], "year": 2015},
    {"title": "Feature Pyramid Networks for Object Detection", "abstract": "Feature pyramids are a basic component in recognition systems for detecting objects at different scales. But recent deep learning object detectors have avoided pyramid representations, in part because they are compute and memory intensive. In this paper, we exploit the inherent multi-scale, pyramidal hierarchy of deep convolutional networks to construct feature pyramids with marginal extra cost.", "categories": ["cs.CV"], "year": 2017},
    {"title": "Mask R-CNN", "abstract": "We present a conceptually simple, flexible, and general framework for object instance segmentation. Our approach efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance. The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition.", "categories": ["cs.CV"], "year": 2017},
    {"title": "U-Net: Convolutional Networks for Biomedical Image Segmentation", "abstract": "There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization.", "categories": ["cs.CV"], "year": 2015},
    {"title": "Semantic Segmentation using Fully Convolutional Networks", "abstract": "Convolutional networks are powerful visual models that yield hierarchies of features. We show that convolutional networks by themselves, trained end-to-end, pixels-to-pixels, exceed the state-of-the-art in semantic segmentation. Our key insight is to build fully convolutional networks that take input of arbitrary size and produce correspondingly-sized output with efficient inference and learning.", "categories": ["cs.CV"], "year": 2015},
    {"title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", "abstract": "While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks.", "categories": ["cs.CV", "cs.LG"], "year": 2021},
    {"title": "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", "abstract": "This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformer from language to vision arise from differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text.", "categories": ["cs.CV"], "year": 2021},
    
    # Generative Models
    {"title": "Generative Adversarial Networks", "abstract": "We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake.", "categories": ["cs.LG", "stat.ML"], "year": 2014},
    {"title": "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks", "abstract": "In recent years, supervised learning with convolutional networks has seen huge adoption in computer vision applications. Comparatively, unsupervised learning with CNNs has received less attention. In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning. We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs).", "categories": ["cs.LG", "cs.CV"], "year": 2016},
    {"title": "Progressive Growing of GANs for Improved Quality, Stability, and Variation", "abstract": "We describe a new training methodology for generative adversarial networks. The key idea is to grow both the generator and discriminator progressively: starting from a low resolution, we add new layers that model increasingly fine details as training progresses. This both speeds the training up and greatly stabilizes it, allowing us to produce images of unprecedented quality.", "categories": ["cs.NE", "cs.LG"], "year": 2018},
    {"title": "A Style-Based Generator Architecture for Generative Adversarial Networks", "abstract": "We propose an alternative generator architecture for generative adversarial networks, borrowing from style transfer literature. The new architecture leads to an automatically learned, unsupervised separation of high-level attributes and stochastic variation in the generated images, and it enables intuitive, scale-specific control of the synthesis.", "categories": ["cs.NE", "cs.LG"], "year": 2019},
    {"title": "Analyzing and Improving the Image Quality of StyleGAN", "abstract": "The style-based GAN architecture (StyleGAN) yields state-of-the-art results in data-driven unconditional generative image modeling. We expose and analyze several of its characteristic artifacts, and propose changes in both model architecture and training methods to address them. In particular, we redesign the generator normalization, revisit progressive growing, and regularize the generator to encourage good conditioning in the mapping from latent codes to images.", "categories": ["cs.CV", "cs.LG"], "year": 2020},
    {"title": "Variational Autoencoders", "abstract": "How can we perform efficient inference and learning in directed probabilistic models, in the presence of continuous latent variables with intractable posterior distributions? We introduce a stochastic variational inference and learning algorithm that scales to large datasets and, under some mild differentiability conditions, even works in the intractable case.", "categories": ["cs.LG", "stat.ML"], "year": 2014},
    {"title": "Denoising Diffusion Probabilistic Models", "abstract": "We present high quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics. Our best results are obtained by training on a weighted variational bound designed according to a novel connection between diffusion probabilistic models and denoising score matching with Langevin dynamics.", "categories": ["cs.LG", "stat.ML"], "year": 2020},
    {"title": "High-Resolution Image Synthesis with Latent Diffusion Models", "abstract": "By decomposing the image formation process into a sequential application of denoising autoencoders, diffusion models achieve state-of-the-art synthesis results on image data and beyond. Additionally, their formulation allows for a guiding mechanism to control the image generation process without retraining. However, since these models typically operate directly in pixel space, optimization of powerful DMs often consumes hundreds of GPU days and inference is expensive due to sequential evaluations.", "categories": ["cs.CV"], "year": 2022},
    {"title": "DALL-E: Zero-Shot Text-to-Image Generation", "abstract": "Text-to-image generation has traditionally focused on finding better modeling assumptions for training on a fixed dataset. These assumptions might involve complex architectures, auxiliary losses, or side information such as object part labels or segmentation masks. We describe a simple approach for this task based on a transformer that autoregressively models the text and image tokens as a single stream of data.", "categories": ["cs.CV", "cs.LG"], "year": 2021},
    
    # Reinforcement Learning
    {"title": "Playing Atari with Deep Reinforcement Learning", "abstract": "We present the first deep learning model to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning. The model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards. We apply our method to seven Atari 2600 games from the Arcade Learning Environment.", "categories": ["cs.LG"], "year": 2013},
    {"title": "Human-level Control Through Deep Reinforcement Learning", "abstract": "The theory of reinforcement learning provides a normative account, deeply rooted in psychological and neuroscientific perspectives on animal behaviour, of how agents may optimize their control of an environment. To use reinforcement learning successfully in situations approaching real-world complexity, however, agents are confronted with a difficult task: they must derive efficient representations of the environment from high-dimensional sensory inputs, and use these to generalize past experience to new situations.", "categories": ["cs.LG"], "year": 2015},
    {"title": "Mastering the Game of Go with Deep Neural Networks and Tree Search", "abstract": "The game of Go has long been viewed as the most challenging of classic games for artificial intelligence owing to its enormous search space and the difficulty of evaluating board positions and moves. Here we introduce a new approach to computer Go that uses value networks to evaluate board positions and policy networks to select moves. These deep neural networks are trained by a novel combination of supervised learning from human expert games, and reinforcement learning from games of self-play.", "categories": ["cs.AI"], "year": 2016},
    {"title": "Mastering the Game of Go without Human Knowledge", "abstract": "A long-standing goal of artificial intelligence is an algorithm that learns, tabula rasa, superhuman proficiency in challenging domains. Recently, AlphaGo became the first program to defeat a world champion in the game of Go. The tree search in AlphaGo evaluated positions and selected moves using deep neural networks. These neural networks were trained by supervised learning from human expert moves, and by reinforcement learning from self-play.", "categories": ["cs.AI"], "year": 2017},
    {"title": "Proximal Policy Optimization Algorithms", "abstract": "We propose a new family of policy gradient methods for reinforcement learning, which alternate between sampling data through interaction with the environment, and optimizing a surrogate objective function using stochastic gradient ascent. Whereas standard policy gradient methods perform one gradient update per data sample, we propose a novel objective function that enables multiple epochs of minibatch updates.", "categories": ["cs.LG"], "year": 2017},
    {"title": "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor", "abstract": "Model-free deep reinforcement learning algorithms have been demonstrated on a range of challenging decision making and control tasks. However, these methods typically suffer from two major challenges: very high sample complexity and brittle convergence properties, which necessitate meticulous hyperparameter tuning. Both of these challenges severely limit the applicability of such methods to complex, real-world domains.", "categories": ["cs.LG", "cs.AI"], "year": 2018},
    {"title": "World Models", "abstract": "We explore building generative neural network models of popular reinforcement learning environments. Our world model can be trained quickly in an unsupervised manner to learn a compressed spatial and temporal representation of the environment. By using features extracted from the world model as inputs to an agent, we can train a very compact and simple policy that can solve the required task.", "categories": ["cs.LG", "cs.NE"], "year": 2018},
    {"title": "Model-Based Reinforcement Learning for Atari", "abstract": "Model-free reinforcement learning methods are known to be effective at learning board and video games and robotics tasks. However, they require enormous numbers of environment interactions. Model-based methods are more sample-efficient but usually require task-specific models and extensive fine-tuning. We demonstrate that model-based methods can achieve similar final performance to model-free approaches.", "categories": ["cs.LG"], "year": 2020},
    
    # Word Embeddings & NLP
    {"title": "Efficient Estimation of Word Representations in Vector Space", "abstract": "We propose two novel model architectures for computing continuous vector representations of words from very large data sets. The quality of these representations is measured in a word similarity task, and the results are compared to the previously best performing techniques based on different types of neural networks. We observe large improvements in accuracy at much lower computational cost.", "categories": ["cs.CL"], "year": 2013},
    {"title": "Distributed Representations of Words and Phrases and their Compositionality", "abstract": "The recently introduced continuous Skip-gram model is an efficient method for learning high-quality distributed vector representations that capture a large number of precise syntactic and semantic word relationships. In this paper we present several extensions that improve both the quality of the vectors and the training speed. By subsampling of the frequent words we obtain significant speedup and also learn more regular word representations.", "categories": ["cs.CL"], "year": 2013},
    {"title": "GloVe: Global Vectors for Word Representation", "abstract": "Recent methods for learning vector space representations of words have succeeded in capturing fine-grained semantic and syntactic regularities using vector arithmetic, but the origin of these regularities has remained opaque. We analyze and make explicit the model properties needed for such regularities to emerge in word vectors. The result is a new global log-bilinear regression model that combines the advantages of the two major model families in the literature.", "categories": ["cs.CL"], "year": 2014},
    {"title": "ELMo: Deep Contextualized Word Representations", "abstract": "We introduce a new type of deep contextualized word representation that models both complex characteristics of word use and how these uses vary across linguistic contexts. Our word vectors are learned functions of the internal states of a deep bidirectional language model, which is pre-trained on a large text corpus. We show that these representations can be easily added to existing models and significantly improve the state of the art.", "categories": ["cs.CL"], "year": 2018},
    {"title": "fastText: Enriching Word Vectors with Subword Information", "abstract": "Continuous word representations, trained on large unlabeled corpora are useful for many natural language processing tasks. Popular models that learn such representations ignore the morphology of words, by assigning a distinct vector to each word. This is a limitation, especially for languages with large vocabularies and many rare words. In this paper, we propose a new approach based on the skipgram model, where each word is represented as a bag of character n-grams.", "categories": ["cs.CL"], "year": 2017},
    
    # Neural Network Fundamentals
    {"title": "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", "abstract": "Deep neural nets with a large number of parameters are very powerful machine learning systems. However, overfitting is a serious problem in such networks. Large networks are also slow to use, making it difficult to deal with overfitting by combining the predictions of many different large neural nets at test time. Dropout is a technique for addressing this problem.", "categories": ["cs.LG"], "year": 2014},
    {"title": "Adam: A Method for Stochastic Optimization", "abstract": "We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters.", "categories": ["cs.LG"], "year": 2015},
    {"title": "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", "abstract": "Training Deep Neural Networks is complicated by the fact that the distribution of each layer's inputs changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rates and careful parameter initialization, and makes it notoriously hard to train models with saturating nonlinearities. We refer to this phenomenon as internal covariate shift, and address the problem by normalizing layer inputs.", "categories": ["cs.LG"], "year": 2015},
    {"title": "Layer Normalization", "abstract": "Training state-of-the-art, deep neural networks is computationally expensive. One way to reduce the training time is to normalize the activities of the neurons. A recently introduced technique called batch normalization uses the distribution of the summed input to a neuron over a mini-batch of training cases to compute a mean and variance which are then used to normalize the summed input to that neuron on each training case.", "categories": ["cs.LG"], "year": 2016},
    {"title": "Group Normalization", "abstract": "Batch Normalization is a milestone technique in the development of deep learning, enabling various networks to train. However, normalizing along the batch dimension introduces problems: BN's error increases rapidly when the batch size becomes smaller, caused by inaccurate batch statistics estimation. This limits BN's usage for training larger models and transferring features to computer vision tasks.", "categories": ["cs.CV", "cs.LG"], "year": 2018},
    {"title": "Rectified Linear Units Improve Restricted Boltzmann Machines", "abstract": "Restricted Boltzmann machines were developed using binary stochastic hidden units. These can be generalized by replacing each binary unit by an infinite number of copies that all have the same weights but have progressively more negative biases. The learning and inference rules for these stepped sigmoid units are unchanged. We show that the negative phase of learning can be made more efficient by using rectified linear units.", "categories": ["cs.NE", "cs.LG"], "year": 2010},
    {"title": "Maxout Networks", "abstract": "We consider the problem of designing models to leverage a recently introduced approximate model averaging technique called dropout. We define a simple new model called maxout that is designed to both facilitate optimization by dropout and improve the accuracy of dropout's fast approximate model averaging technique.", "categories": ["cs.LG", "stat.ML"], "year": 2013},
    {"title": "Deep Learning", "abstract": "Deep learning allows computational models that are composed of multiple processing layers to learn representations of data with multiple levels of abstraction. These methods have dramatically improved the state-of-the-art in speech recognition, visual object recognition, object detection and many other domains such as drug discovery and genomics.", "categories": ["cs.LG"], "year": 2015},
    {"title": "Understanding the Difficulty of Training Deep Feedforward Neural Networks", "abstract": "Whereas before 2006 it appears that deep multi-layer neural networks were not successfully trained, since then several algorithms have been shown to successfully train them, with experimental results showing the superiority of deeper vs less deep architectures. All these experimental results were obtained with new initialization or training mechanisms. Our objective here is to understand better why standard gradient descent from random initialization is doing so poorly with deep neural networks.", "categories": ["cs.LG"], "year": 2010},
    {"title": "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification", "abstract": "Rectified activation units are essential for state-of-the-art neural networks. In this work, we study rectifier neural networks for image classification from two aspects. First, we propose a Parametric Rectified Linear Unit that generalizes the traditional rectified unit. PReLU improves model fitting with nearly zero extra computational cost and little overfitting risk.", "categories": ["cs.CV", "cs.LG"], "year": 2015},
    
    # Sequence Models
    {"title": "Long Short-Term Memory", "abstract": "Learning to store information over extended time intervals by recurrent backpropagation takes a very long time, mostly because of insufficient, decaying error backflow. We briefly review Hochreiter's 1991 analysis of this problem, then address it by introducing a novel, efficient, gradient-based method called long short-term memory (LSTM). Truncating the gradient where this does not do harm, LSTM can learn to bridge minimal time lags in excess of 1000 discrete-time steps.", "categories": ["cs.NE"], "year": 1997},
    {"title": "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation", "abstract": "In this paper, we propose a novel neural network model called RNN Encoder-Decoder that consists of two recurrent neural networks. One RNN encodes a sequence of symbols into a fixed-length vector representation, and the other decodes the representation into another sequence of symbols. The encoder and decoder of the proposed model are jointly trained to maximize the conditional probability of a target sequence given a source sequence.", "categories": ["cs.CL", "cs.LG"], "year": 2014},
    {"title": "Sequence to Sequence Learning with Neural Networks", "abstract": "Deep Neural Networks are powerful models that have achieved excellent performance on difficult learning tasks. Although DNNs work well whenever large labeled training sets are available, they cannot be used to map sequences to sequences. In this paper, we present a general end-to-end approach to sequence learning that makes minimal assumptions on the sequence structure.", "categories": ["cs.CL"], "year": 2014},
    {"title": "Neural Machine Translation by Jointly Learning to Align and Translate", "abstract": "Neural machine translation is a recently proposed approach to machine translation. Unlike the traditional statistical machine translation, the neural machine translation aims at building a single neural network that can be jointly tuned to maximize the translation performance. The models proposed recently for neural machine translation often belong to a family of encoder-decoders and encode a source sentence into a fixed-length vector from which a decoder generates a translation.", "categories": ["cs.CL"], "year": 2015},
    {"title": "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling", "abstract": "In this paper we compare different types of recurrent units in recurrent neural networks. Especially, we focus on more sophisticated units that implement a gating mechanism, such as a long short-term memory unit and a recently proposed gated recurrent unit. We evaluate these recurrent units on the tasks of polyphonic music modeling and speech signal modeling.", "categories": ["cs.NE", "cs.LG"], "year": 2014},
    {"title": "Recurrent Neural Network based Language Model", "abstract": "A new recurrent neural network based language model with applications to speech recognition is presented. Results indicate that it is possible to obtain significant improvements in perplexity over standard n-gram language models, and improvements in speech recognition accuracy. The model uses classes to compress the neural network output layer, which significantly speeds up training.", "categories": ["cs.CL"], "year": 2010},
    
    # Graph Neural Networks
    {"title": "Semi-Supervised Classification with Graph Convolutional Networks", "abstract": "We present a scalable approach for semi-supervised learning on graph-structured data that is based on an efficient variant of convolutional neural networks which operate directly on graphs. We motivate the choice of our convolutional architecture via a localized first-order approximation of spectral graph convolutions. Our model scales linearly in the number of graph edges.", "categories": ["cs.LG", "stat.ML"], "year": 2017},
    {"title": "Graph Attention Networks", "abstract": "We present graph attention networks (GATs), novel neural network architectures that operate on graph-structured data, leveraging masked self-attentional layers to address the shortcomings of prior methods based on graph convolutions or their approximations. By stacking layers in which nodes are able to attend over their neighborhoods' features, we enable specifying different weights to different nodes in a neighborhood.", "categories": ["cs.LG", "stat.ML"], "year": 2018},
    {"title": "Inductive Representation Learning on Large Graphs", "abstract": "Low-dimensional embeddings of nodes in large graphs have proved extremely useful in a variety of prediction tasks, from content recommendation to identifying protein functions. However, most existing approaches require that all nodes in the graph are present during training of the embeddings; these previous approaches are inherently transductive and do not naturally generalize to unseen nodes.", "categories": ["cs.LG", "cs.SI"], "year": 2017},
    {"title": "How Powerful are Graph Neural Networks?", "abstract": "Graph Neural Networks (GNNs) are an effective framework for representation learning of graphs. GNNs follow a neighborhood aggregation scheme, where the representation vector of a node is computed by recursively aggregating and transforming representation vectors of its neighboring nodes. Many GNN variants have been proposed and have achieved state-of-the-art results on both node and graph classification tasks.", "categories": ["cs.LG", "cs.SI"], "year": 2019},
    {"title": "Neural Message Passing for Quantum Chemistry", "abstract": "Supervised learning on molecules has incredible potential to be useful in chemistry, drug discovery, and materials science. Luckily, several promising and closely related neural network models have emerged in recent years for predicting properties of molecules. We attempt to unify these models into a single framework called Message Passing Neural Networks.", "categories": ["cs.LG", "physics.chem-ph"], "year": 2017},
    {"title": "Graph Neural Networks: A Review of Methods and Applications", "abstract": "Lots of learning tasks require dealing with graph data which contains rich relation information among elements. Modeling physics systems, learning molecular fingerprints, predicting protein interface, and classifying diseases all require a model to learn from graph inputs. In recent years, Graph Neural Networks have become a widely used graph analysis method due to their convincing performance and high interpretability.", "categories": ["cs.LG"], "year": 2019},
    
    # Information Retrieval & Search
    {"title": "BERT for Information Retrieval: A Survey", "abstract": "Pre-trained language models such as BERT have achieved state-of-the-art results across many NLP tasks. Information retrieval is one of the important application areas that has been transformed by these models. In this survey, we discuss how BERT has been applied to various IR tasks including ad-hoc retrieval, question answering, and conversational search.", "categories": ["cs.IR", "cs.CL"], "year": 2020},
    {"title": "Dense Passage Retrieval for Open-Domain Question Answering", "abstract": "Open-domain question answering relies on efficient passage retrieval to select candidate contexts, where traditional sparse vector space models, such as TF-IDF or BM25, are the de facto method. In this work, we show that retrieval can be practically implemented using dense representations alone, where embeddings are learned from a small number of questions and passages by a simple dual-encoder framework.", "categories": ["cs.CL", "cs.IR"], "year": 2020},
    {"title": "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT", "abstract": "Recent progress in Natural Language Understanding (NLU) is driving fast-paced advances in Information Retrieval (IR), largely owed to fine-tuning deep language models for document ranking. While remarkably effective, the ranking models based on these large pretrained transformers are particularly expensive, often requiring latencies that are orders of magnitude larger than traditional bag-of-words models.", "categories": ["cs.IR"], "year": 2020},
    {"title": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", "abstract": "BERT and RoBERTa have set a new state-of-the-art performance on sentence-pair regression tasks like semantic textual similarity. However, it requires that both sentences are fed into the network, which causes a massive computational overhead. In this publication, we present Sentence-BERT (SBERT), a modification of the pretrained BERT network that use siamese and triplet network structures to derive semantically meaningful sentence embeddings.", "categories": ["cs.CL"], "year": 2019},
    {"title": "Learning to Rank: From Pairwise Approach to Listwise Approach", "abstract": "The learning to rank problem has attracted significant attention from both the information retrieval and machine learning communities. In this paper, we first introduce the learning to rank problem along with its applications. We then describe existing approaches to the problem, including the pointwise, pairwise, and listwise approaches. We also introduce the representative algorithms in each approach.", "categories": ["cs.IR", "cs.LG"], "year": 2007},
    {"title": "Neural Information Retrieval: A Literature Review", "abstract": "This paper surveys research advances in neural information retrieval (neural IR) from the perspective of modeling and training objectives. Neural IR models can be broadly categorized into representation learning and interaction models. Representation learning focuses on learning continuous representations for queries and documents independently, enabling efficient retrieval via approximate nearest neighbor search.", "categories": ["cs.IR"], "year": 2018},
    {"title": "Pre-training Methods in Information Retrieval", "abstract": "Pre-trained language models have led to a paradigm shift in information retrieval. This paper provides a comprehensive review of pre-training methods for information retrieval. We organize our discussion around three dimensions: 1) the pre-training corpus, 2) the pre-training objectives, and 3) the model architectures. We also discuss downstream tasks and evaluation methodologies.", "categories": ["cs.IR", "cs.CL"], "year": 2022},
    
    # Miscellaneous Important Papers
    {"title": "XGBoost: A Scalable Tree Boosting System", "abstract": "Tree boosting is a highly effective and widely used machine learning method. In this paper, we describe a scalable end-to-end tree boosting system called XGBoost, which is used widely by data scientists to achieve state-of-the-art results on many machine learning challenges. We propose a novel sparsity-aware algorithm for sparse data and weighted quantile sketch for approximate tree learning.", "categories": ["cs.LG"], "year": 2016},
    {"title": "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", "abstract": "Gradient Boosting Decision Tree is a popular machine learning algorithm, and has quite a few effective implementations such as XGBoost and pGBRT. Although many engineering optimizations have been adopted in these implementations, the efficiency and scalability are still unsatisfactory when the feature dimension is high and data size is large.", "categories": ["cs.LG"], "year": 2017},
    {"title": "Random Forests", "abstract": "Random forests are a combination of tree predictors such that each tree depends on the values of a random vector sampled independently and with the same distribution for all trees in the forest. The generalization error for forests converges a.s. to a limit as the number of trees in the forest becomes large.", "categories": ["cs.LG", "stat.ML"], "year": 2001},
    {"title": "Support Vector Machine", "abstract": "The support vector machine is a learning machine for two-group classification problems. The machine conceptually implements the following idea: input vectors are non-linearly mapped to a very high-dimensional feature space. In this feature space a linear decision surface is constructed. Special properties of the decision surface ensure high generalization ability of the learning machine.", "categories": ["cs.LG"], "year": 1995},
    {"title": "k-Nearest Neighbor Algorithm", "abstract": "The k-nearest neighbor algorithm is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. The output depends on whether k-NN is used for classification or regression. In k-NN classification, the output is a class membership.", "categories": ["cs.LG"], "year": 1992},
    {"title": "Principal Component Analysis", "abstract": "Principal component analysis is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. This transformation is defined in such a way that the first principal component has the largest possible variance.", "categories": ["stat.ML", "cs.LG"], "year": 1901},
    {"title": "t-SNE: Visualizing Data using t-Distributed Stochastic Neighbor Embedding", "abstract": "We present a new technique called t-SNE that visualizes high-dimensional data by giving each datapoint a location in a two or three-dimensional map. The technique is a variation of Stochastic Neighbor Embedding that is much easier to optimize, and produces significantly better visualizations by reducing the tendency to crowd points together in the center of the map.", "categories": ["cs.LG", "stat.ML"], "year": 2008},
    {"title": "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction", "abstract": "UMAP (Uniform Manifold Approximation and Projection) is a novel manifold learning technique for dimension reduction. UMAP is constructed from a theoretical framework based in Riemannian geometry and algebraic topology. The result is a practical scalable algorithm that applies to real world data.", "categories": ["cs.LG", "stat.ML"], "year": 2018},
    {"title": "Mean Shift: A Robust Approach Toward Feature Space Analysis", "abstract": "A general non-parametric technique is proposed for the analysis of a complex multimodal feature space and to delineate arbitrarily shaped clusters in it. The basic computational module of the technique is an old pattern recognition procedure, the mean shift. We prove for discrete data the convergence of a recursive mean shift procedure to the nearest stationary point of the underlying density function.", "categories": ["cs.CV"], "year": 2002},
    {"title": "DBSCAN: A Density-Based Algorithm for Discovering Clusters", "abstract": "Clustering algorithms are attractive for the task of class identification in spatial databases. However, the application to large spatial databases rises the following requirements for clustering algorithms: minimal requirements of domain knowledge to determine the input parameters, discovery of clusters with arbitrary shape and good efficiency on large databases.", "categories": ["cs.DB", "cs.LG"], "year": 1996},
]


def expand_corpus(base_papers: list, target_size: int = 1000) -> list:
    """
    Expand the corpus by generating variations of existing papers.
    Creates realistic variations while maintaining diversity.
    """
    import hashlib
    
    expanded = []
    
    # First add all base papers
    for i, paper in enumerate(base_papers):
        expanded.append({
            "doc_id": f"arxiv_{i:04d}",
            "title": paper["title"],
            "abstract": paper["abstract"],
            "authors": generate_authors(paper["title"]),
            "categories": paper["categories"],
            "year": paper["year"]
        })
    
    # Topic variations for generating more papers
    topic_variations = {
        "cs.CL": [
            ("language model", ["pre-trained", "fine-tuned", "multilingual", "low-resource", "domain-specific"]),
            ("text classification", ["sentiment analysis", "topic modeling", "intent detection", "spam detection"]),
            ("named entity recognition", ["sequence labeling", "token classification", "information extraction"]),
            ("machine translation", ["neural machine translation", "multilingual translation", "low-resource translation"]),
            ("question answering", ["reading comprehension", "open-domain QA", "conversational QA"]),
            ("summarization", ["abstractive summarization", "extractive summarization", "multi-document summarization"]),
            ("natural language understanding", ["semantic parsing", "natural language inference", "textual entailment"]),
        ],
        "cs.CV": [
            ("image classification", ["fine-grained recognition", "multi-label classification", "zero-shot recognition"]),
            ("object detection", ["anchor-free detection", "one-stage detection", "two-stage detection"]),
            ("semantic segmentation", ["instance segmentation", "panoptic segmentation", "scene parsing"]),
            ("image generation", ["image synthesis", "image-to-image translation", "super-resolution"]),
            ("video understanding", ["action recognition", "video classification", "temporal modeling"]),
            ("face recognition", ["face verification", "face identification", "facial attribute analysis"]),
            ("pose estimation", ["human pose estimation", "3D pose estimation", "hand pose estimation"]),
        ],
        "cs.LG": [
            ("neural network", ["deep learning", "representation learning", "feature learning"]),
            ("optimization", ["gradient descent", "adaptive optimization", "hyperparameter optimization"]),
            ("regularization", ["dropout", "weight decay", "data augmentation"]),
            ("transfer learning", ["domain adaptation", "multi-task learning", "meta-learning"]),
            ("self-supervised learning", ["contrastive learning", "pretext tasks", "representation learning"]),
            ("federated learning", ["distributed learning", "privacy-preserving learning", "decentralized learning"]),
            ("neural architecture search", ["AutoML", "architecture optimization", "efficient neural networks"]),
        ],
        "cs.AI": [
            ("knowledge representation", ["knowledge graphs", "ontologies", "semantic networks"]),
            ("reasoning", ["logical reasoning", "commonsense reasoning", "causal reasoning"]),
            ("planning", ["task planning", "motion planning", "sequential decision making"]),
            ("multi-agent systems", ["cooperative agents", "competitive agents", "emergent behavior"]),
        ],
        "cs.IR": [
            ("information retrieval", ["document retrieval", "passage retrieval", "semantic search"]),
            ("ranking", ["learning to rank", "neural ranking", "relevance estimation"]),
            ("recommendation", ["collaborative filtering", "content-based filtering", "hybrid methods"]),
            ("query understanding", ["query expansion", "query reformulation", "intent classification"]),
        ],
    }
    
    # Generate additional papers with balanced categories
    paper_id = len(base_papers)
    random.seed(42)
    
    # Ensure balanced distribution
    category_weights = {"cs.CL": 0.25, "cs.CV": 0.25, "cs.LG": 0.30, "cs.AI": 0.10, "cs.IR": 0.10}
    categories_list = list(category_weights.keys())
    weights = list(category_weights.values())
    
    while len(expanded) < target_size:
        # Pick a category based on weights
        category = random.choices(categories_list, weights=weights, k=1)[0]
        
        if category not in topic_variations:
            continue
            
        topic_group = random.choice(topic_variations[category])
        base_topic, variations = topic_group
        variation = random.choice(variations)
        
        # Generate title
        title_templates = [
            f"Improving {variation.title()} with Novel Architecture",
            f"A Study of {variation.title()} Methods",
            f"Efficient {variation.title()} using Deep Learning",
            f"Towards Better {variation.title()}: A Comparative Analysis",
            f"{variation.title()}: Challenges and Opportunities",
            f"Learning {variation.title()} from Limited Data",
            f"Scalable {variation.title()} with Attention Mechanisms",
            f"Self-Supervised Approaches for {variation.title()}",
            f"Multi-Task Learning for {variation.title()}",
            f"Robust {variation.title()} under Distribution Shift",
        ]
        
        title = random.choice(title_templates)
        
        # Generate abstract with keywords that match the topic
        abstract_templates = [
            f"We propose a novel approach to {variation} that achieves state-of-the-art results on standard benchmarks. Our method leverages recent advances in {base_topic} and deep learning to address key challenges. Extensive experiments demonstrate the effectiveness of our approach, showing significant improvements over existing methods. We provide detailed ablation studies and analysis to understand the contribution of each component.",
            f"This paper presents a comprehensive study of {variation} methods for {base_topic}. We systematically evaluate various approaches and identify key factors that contribute to performance. Our analysis reveals important insights about the relationship between model architecture and task performance. Based on these findings, we propose several improvements that lead to better results.",
            f"We introduce a new framework for {variation} that combines the strengths of multiple existing approaches. Our method addresses the {base_topic} problem and is designed to be efficient and scalable, making it suitable for large-scale applications. We demonstrate the effectiveness of our approach through extensive experiments on multiple datasets.",
            f"In this work, we address the problem of {variation} from a new perspective. We propose a learning-based approach for {base_topic} that automatically adapts to the characteristics of the input data. Our method achieves robust performance across diverse scenarios and generalizes well to unseen domains.",
            f"We present a detailed investigation of {variation} techniques and their applications to {base_topic}. Our study covers both classical and neural approaches, providing a unified view of the field. We identify key challenges and propose solutions that improve the state of the art.",
        ]
        
        abstract = random.choice(abstract_templates)
        
        # Secondary category
        secondary_cats = [c for c in categories_list if c != category]
        secondary = random.choice(secondary_cats) if random.random() > 0.5 else None
        
        cats = [category]
        if secondary:
            cats.append(secondary)
        
        # Create paper entry
        expanded.append({
            "doc_id": f"arxiv_{paper_id:04d}",
            "title": title,
            "abstract": abstract,
            "authors": generate_authors(title),
            "categories": cats,
            "year": random.randint(2018, 2024)
        })
        
        paper_id += 1
    
    return expanded[:target_size]


def generate_authors(title: str) -> list:
    """Generate plausible author names."""
    first_names = ["Wei", "Jing", "Ming", "Xiao", "Yu", "Chen", "Li", "Zhang", "Yang", "Wang",
                   "John", "Michael", "David", "James", "Robert", "Sarah", "Emily", "Jennifer",
                   "Alexander", "Maria", "Thomas", "Anna", "Peter", "Laura", "Martin"]
    last_names = ["Zhang", "Li", "Wang", "Chen", "Liu", "Yang", "Huang", "Wu", "Zhou", "Xu",
                  "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Wilson",
                  "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris"]
    
    random.seed(hash(title) % 2**32)
    num_authors = random.randint(2, 5)
    authors = []
    for _ in range(num_authors):
        first = random.choice(first_names)
        last = random.choice(last_names)
        authors.append(f"{first} {last}")
    return authors


def generate_queries(corpus: list, num_queries: int = 30) -> list:
    """Generate evaluation queries with relevance judgments."""
    random.seed(42)
    
    queries = []
    
    # Manually curated high-quality queries with known relevant papers
    curated_queries = [
        {"query": "transformer attention mechanism", "keywords": ["transformer", "attention"]},
        {"query": "image classification convolutional network", "keywords": ["image", "classification", "convolutional"]},
        {"query": "object detection real time", "keywords": ["object", "detection"]},
        {"query": "word embeddings representation", "keywords": ["word", "embedding", "representation"]},
        {"query": "generative adversarial network", "keywords": ["generative", "adversarial", "gan"]},
        {"query": "reinforcement learning game", "keywords": ["reinforcement", "learning", "game"]},
        {"query": "recurrent neural network sequence", "keywords": ["recurrent", "sequence", "lstm"]},
        {"query": "graph neural network", "keywords": ["graph", "neural", "network"]},
        {"query": "language model pre-training", "keywords": ["language", "model", "pre-train"]},
        {"query": "machine translation neural", "keywords": ["translation", "neural", "machine"]},
        {"query": "image segmentation semantic", "keywords": ["segmentation", "semantic", "image"]},
        {"query": "optimization gradient descent", "keywords": ["optimization", "gradient"]},
        {"query": "dropout regularization", "keywords": ["dropout", "regularization"]},
        {"query": "batch normalization training", "keywords": ["batch", "normalization"]},
        {"query": "residual learning deep network", "keywords": ["residual", "deep"]},
        {"query": "passage retrieval question answering", "keywords": ["retrieval", "passage", "question"]},
        {"query": "sentence embeddings similarity", "keywords": ["sentence", "embedding", "similarity"]},
        {"query": "diffusion model generation", "keywords": ["diffusion", "generation"]},
        {"query": "vision transformer", "keywords": ["vision", "transformer"]},
        {"query": "knowledge distillation", "keywords": ["knowledge", "distillation"]},
        {"query": "contrastive learning self-supervised", "keywords": ["contrastive", "self-supervised"]},
        {"query": "few-shot learning", "keywords": ["few-shot", "meta"]},
        {"query": "text classification sentiment", "keywords": ["text", "classification", "sentiment"]},
        {"query": "named entity recognition", "keywords": ["named", "entity", "recognition"]},
        {"query": "information retrieval ranking", "keywords": ["information", "retrieval", "ranking"]},
        {"query": "tree boosting gradient", "keywords": ["boosting", "tree", "gradient"]},
        {"query": "dimensionality reduction", "keywords": ["dimensionality", "reduction"]},
        {"query": "clustering algorithm", "keywords": ["clustering", "cluster"]},
        {"query": "mobile efficient network", "keywords": ["mobile", "efficient"]},
        {"query": "policy gradient reinforcement", "keywords": ["policy", "gradient", "reinforcement"]},
        {"query": "domain adaptation transfer", "keywords": ["domain", "adaptation", "transfer"]},
        {"query": "multi-task learning", "keywords": ["multi-task", "learning"]},
        {"query": "neural architecture search", "keywords": ["architecture", "search", "neural"]},
        {"query": "face recognition verification", "keywords": ["face", "recognition"]},
        {"query": "video action recognition", "keywords": ["video", "action"]},
    ]
    
    stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
                 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
                 'these', 'those', 'it', 'its', 'we', 'our', 'their', 'which', 'what', 'who', 'using',
                 'based', 'via', 'paper', 'propose', 'proposed', 'show', 'approach', 'method', 'methods',
                 'results', 'model', 'models', 'new', 'novel', 'present', 'introduce'}
    
    for i, q in enumerate(curated_queries[:num_queries]):
        keywords = set(q["keywords"])
        query_words = set(q["query"].lower().split()) - stopwords
        
        relevant = []
        for paper in corpus:
            paper_text = f"{paper['title']} {paper['abstract']}".lower()
            
            # Check keyword match
            keyword_matches = sum(1 for kw in keywords if kw in paper_text)
            word_matches = sum(1 for w in query_words if w in paper_text)
            
            total_match = keyword_matches + word_matches
            
            if total_match >= 2:
                if keyword_matches >= 2 or total_match >= 4:
                    score = 3  # High relevance
                elif total_match >= 3:
                    score = 2  # Medium relevance
                else:
                    score = 1  # Low relevance
                relevant.append((paper["doc_id"], score))
        
        # Sort by score and take top results
        relevant.sort(key=lambda x: -x[1])
        relevant = relevant[:10]
        
        if len(relevant) >= 1:
            queries.append({
                "query_id": f"q{len(queries)+1:02d}",
                "query_text": q["query"],
                "relevant_docs": [doc_id for doc_id, _ in relevant],
                "graded_relevance": {doc_id: score for doc_id, score in relevant}
            })
    
    return queries


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create ArXiv-style ML papers corpus")
    parser.add_argument('--num-papers', type=int, default=1000, help='Number of papers')
    parser.add_argument('--num-queries', type=int, default=30, help='Number of queries')
    parser.add_argument('--output-dir', type=str, default='data', help='Output directory')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("CREATING ML/AI RESEARCH PAPERS CORPUS")
    print("=" * 60)
    
    # Create corpus
    print(f"\nGenerating {args.num_papers} papers...")
    corpus = expand_corpus(PAPERS_DATA, target_size=args.num_papers)
    
    # Save corpus
    corpus_path = os.path.join(args.output_dir, 'corpus.json')
    with open(corpus_path, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
    print(f"Saved corpus to: {corpus_path}")
    
    # Generate queries
    print(f"\nGenerating {args.num_queries} evaluation queries...")
    queries = generate_queries(corpus, num_queries=args.num_queries)
    
    # Save queries
    queries_path = os.path.join(args.output_dir, 'queries.json')
    with open(queries_path, 'w', encoding='utf-8') as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(queries)} queries to: {queries_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("CORPUS READY")
    print("=" * 60)
    print(f"Total papers: {len(corpus)}")
    print(f"Test queries: {len(queries)}")
    
    # Category distribution
    cat_counts = {}
    for paper in corpus:
        for cat in paper.get('categories', []):
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
    
    print(f"\nCategory distribution:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cat}: {count}")
    
    # Year distribution
    years = [p['year'] for p in corpus if p.get('year')]
    if years:
        print(f"\nYear range: {min(years)} - {max(years)}")
    
    print("\n" + "=" * 60)
    print("Run evaluation: python main.py --mode evaluate")
    print("=" * 60)


if __name__ == "__main__":
    main()
