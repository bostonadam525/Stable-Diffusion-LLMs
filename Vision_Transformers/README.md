# Vision Transformers
* First vision transformer paper from 2021: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

## Overview of Vision Transformers
* Excellent review post by Pinecone blog: https://www.pinecone.io/learn/series/image-search/vision-transformers/
* Steps/Process:

1. Split the image into image patches.
2. Process patches through the linear projection layer to get initial patch embeddings.
3. Preappend trainable “class” embedding to patch embeddings.
4. Sum patch embeddings and learned positional embeddings.
 
* From pinecone post, a very simple way to think about Image Patches:

![image](https://github.com/user-attachments/assets/815336bf-b1fc-46eb-bc7b-890a55d327b5)

* Similar to how BERT learnes positional encodings:

![image](https://github.com/user-attachments/assets/ac9046eb-0531-44f2-a457-0a243624a6c5)


* Original vision transformer model from original paper: 
![image](https://github.com/user-attachments/assets/d7bdd3e2-6955-4187-87dc-06d9eecc05e8)


* They demonstrated that reliance on a CNN network is NOT necessary. 
* A pure transformer applied directly to image patch sequences performs just as well and even better when fine-tuned.
* This also requires significantly less computation resources.
* Initially trained with small dataset —> foudn that more data produced better results. 
* Standard transformer takes 1D input. Had to transform matrices to handle 2D inputs. 
* ViT-L/16 means the “Large” variant with 16×16 input patch size.


## Main concept of the Vision Transformer
* Quote from the original paper:
* *"Inspired by the Transformer scaling successes in NLP, we experiment with applying a standard Transformer directly to images, with the fewest possible modifications. To do so, we split an image into patches and provide the sequence of linear embeddings of these patches as an input to a Transformer. Image patches are treated the same way as tokens (words) in an NLP application. We train the model on image classification in supervised fashion.”*

* CNNs are known to have a large number of parameters more than a transformer, so size and computational power is helpful with this method.


## Local vs. Global Attention Mechanism
* CNNs primarily extract local features by sliding filters over small image regions, while position global embeddings enable a model to understand the relationship between different image parts across the entire image by incorporating positional information into the embedding process.


## Position Embeddings for Vision Transformers
* This concept is what makes the Vision Transformer more powerful than a standard CNN.
* See this excellent blog post for more details and code implementation: https://towardsdatascience.com/position-embeddings-for-vision-transformers-explained-a6f9add341d5

## Architecture:
* CNNs utilize convolutional layers to extract local features, while models using position global embeddings typically rely on Transformer architecture with self-attention layers to capture global dependencies.

## Application:
* CNNs are widely used for image classification tasks where identifying local features is crucial, while models with position global embeddings are becoming increasingly popular for tasks where understanding the overall context within an image is important

### Example:
1. CNN
  * When classifying an image of a cat, a CNN might focus on identifying the shape of the ears, eyes, and nose, which are local features within the image. 

2. Position Global Embeddings
* A model using position global embeddings would not only identify these local features but also understand the relative positions of these features within the entire image, allowing it to better distinguish the cat from other objects in the background. 

## Vision Transfomers Takeaways
1. ALOT FASTER than CNNs due to less parameters
2. Attention mechanism differs
  * CNN attention is local
  * Vision Transformer —> Attention is Global and uses positional embeddings!
        * Position embedding cosine similarity is key


# Advantages of Vision Transformers
* [source](https://medium.com/@danushidk507/vision-transformers-an-alternative-to-cnn-b1bb620b5c96)
1. **Global Context Understanding**
  * Self-Attention Mechanism
      * Vision Transformers use self-attention to compute relationships between all pairs of patches in an image.
      * This allows the model to capture global dependencies and context, which is challenging for traditional CNNs that primarily focus on local features through convolutional operations.

  * Multi-Head Attention
      * using multi-head attention, ViTs can focus on different parts of the image simultaneously, learning a richer set of features and representations.

2. **Scalability**
  * Model Size
     * Transformers can be easily scaled up by increasing the number of layers, attention heads, and embedding dimensions. This scalability has been demonstrated effectively with NLP models like BERT and GPT, and the same principle applies to ViTs.
  * Training Efficiency
    * With sufficient computational resources and large datasets, ViTs can achieve state-of-the-art performance. The parallel nature of Transformer architecture also allows for efficient training on modern hardware like GPUs and TPUs.

3. Flexibility in Handling Input
  * No Assumption of Locality: Unlike CNNs, which are biased towards local feature extraction due to the convolution operation, ViTs do not assume any locality.
  * This makes them more flexible in capturing long-range dependencies in images.
Adaptability to Different Tasks: ViTs can be easily adapted to various computer vision tasks beyond image classification, such as object detection, segmentation, and image generation, by modifying the architecture and training approach.

4. Improved Performance with Large Datasets
  * Pre-training and Fine-tuning: Similar to NLP, ViTs benefit significantly from large-scale pre-training followed by fine-tuning on specific tasks. This transfer learning approach helps in achieving better performance even with limited task-specific data.
  * Data Augmentation: Techniques like data augmentation and regularization can further enhance the performance of ViTs, making them robust to variations in input data.

5. Potential for Reduced Inductive Bias:
  * Learned Representations: ViTs rely more on the data to learn relevant features and less on handcrafted features or inductive biases. This can lead to discovering novel patterns and representations that might be overlooked by traditional CNNs.


# Applications of Vision Transformers

1. Image Classification
  * Benchmark Performance: ViTs have shown competitive or superior performance compared to state-of-the-art CNNs on benchmark datasets like ImageNet.
  * Transfer Learning: Pre-trained ViTs can be fine-tuned on specific image classification tasks, leveraging the learned representations for improved accuracy.

2. Object Detection
  * DEtection TRansformers (DETR): Vision Transformers have been successfully applied to object detection tasks. Models like DETR use a Transformer-based approach to predict object bounding boxes and classes, achieving robust performance with simpler architectures compared to traditional methods.
  * End-to-End Learning: ViTs enable end-to-end learning for object detection, eliminating the need for hand-crafted anchors and proposals.

3. Image Segmentation
  * Semantic Segmentation - Vision Transformers can be extended for semantic segmentation tasks by incorporating decoder structures that output pixel-level classifications.
  * Instance Segmentation: ViTs can also be adapted for instance segmentation, where individual objects within an image are identified and segmented.

4. Generative Modeling
  * Image Synthesis: ViTs can be used in generative models to create high-quality images. By conditioning on various inputs, they can generate images with specific attributes or styles.
  * Video Generation: Extending ViTs to the temporal domain allows for applications in video generation and editing, capturing both spatial and temporal dependencies.

5. Medical Imaging
  * Diagnosis and Analysis: Vision Transformers are increasingly being applied to medical imaging tasks such as disease diagnosis, tumor detection, and segmentation of medical scans (e.g., MRI, CT).
  * Cross-Modality Learning: ViTs can learn from multiple imaging modalities, enhancing diagnostic accuracy by integrating information from different sources.

6. Remote Sensing and Geospatial Analysis
  * Satellite Imagery: ViTs are used in analyzing satellite images for applications such as land cover classification, change detection, and environmental monitoring.
  * Urban Planning: High-resolution geospatial data processed by ViTs can aid in urban planning and infrastructure development.

7. Autonomous Driving
  * Perception Systems: ViTs are being explored for use in perception systems of autonomous vehicles, improving object detection, lane detection, and scene understanding.
  * Sensor Fusion: Combining data from multiple sensors (cameras, LiDAR, radar) with ViTs can enhance the robustness and accuracy of autonomous driving systems.

8. Fine-Grained Visual Recognition
  * Detailed Classification: ViTs are well-suited for fine-grained visual recognition tasks where distinguishing between subtle differences is crucial, such as species identification in biodiversity studies or product recognition in retail.

9. Robotics and Human-Computer Interaction
  * Vision for Robots: ViTs can improve the visual perception capabilities of robots, enabling better interaction with their environment and humans.
  * Gesture Recognition: Applications in human-computer interaction include using ViTs for accurate gesture and pose recognition.
