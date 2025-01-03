# Basics of Stable Diffusion
* Stable diffusion models form the backbone of text to image models.
* This repo is devoted to an overview of the intuition and understanding of these models.


## High-Level Overview
* Dogma: "Stable diffusion works by converting text to image".
* Reality: Stable diffusion works by converting noise to image via the guidance of text by performing reverse diffusion!


### The 3 important concepts:
1. Noise into image —> this is the most important concept!
2. Via the Guidance of Text
3. Reverse Diffusion

- Audio, Video and other artifacts can be generated besides images. 
- We call this “multi-modal” modeling 


* How this works
1. Text prompt encoded and Sent to Stable Diffusion model. 
2. Reverse diffusion process takes place via “denoising” or removing noise. 
3. Decoding to form an image, audio, video, etc.


## Examples of Basic Stable Diffusion

1. “Guidance”
  * text input —> “Boy” 
  * generates an image of a “boy animal with mouse ears”
  * The text “Boy” guides you to change the “ears”.
    * **The takeaway here is that in Stable Diffusion this guidance is at a Pixel level.**
    * **Encoded text tells us which pixels we need to change by specific amounts. In other words the encoded text guides the Stable Diffusion process.**
   
![image](https://github.com/user-attachments/assets/42740b9c-c688-4100-8a2d-177974926d48)


# Reverse Diffusion
* This is the Central concept of Stable Diffusion!

## Diffusion
* Yes, this is the same thing as molecular diffusion in science or biology where molecules move from an area of higher concentration to an area of lower concentration to achieve **equilibrium**.

## Diffusion in terms of Images
* A diffused image can be imagined as a **True Noisy Image**
* After we fully revert the diffused noise and reconstruct it we have an image described by Text.
* Example of a **True Noisy Image** or in Computer Vision it is known as a **Gaussian Noisy Image**.
  * What this means is that all the pixels are "fully diffused" and completely randomly distributed. 

![image](https://github.com/user-attachments/assets/4b8af846-c99a-47f4-a4a8-752a03917d92)


## Reverse Diffusion Process
* The main concept is that we slightly modify the pixels of a noisy image to make it more similar to the input text or prompt. 


1. Start with a fully diffused image or "True Noisy Image"
2. Denoise the image (many times) --> **Reverse diffusion is also called "Denoising"**
   * Multiple iterations of noise removal.
   * Multiple steps and processes to do this. 
3. Final image result


![image](https://github.com/user-attachments/assets/bad3e008-8a65-4141-9aa1-78e8d96c6618)


### Forward Diffusion (add noise!)
* This is the first step in the Reverse Diffusion process.
* You simply take an image and add **noise** to it.
* The obvious reverse of this (reverse diffusion) is the **denoising** process.
* Below, it is important to understand the terminology:
   * `X0` --> image you want to denoise
   * `XT` --> image with added noise T times or at Step T (time steps)

![image](https://github.com/user-attachments/assets/bc33fa20-acc7-4740-8ae6-05bb9d04566e)



# Purpose of Stable Diffusion
* Start with **noisy image**
* Neural Network intermediary
   * Neural network "predicts the noise" in the image
   * Stable Diffusion equation: `C => R => R`
   * Reverse diffusion by the Neural Network, where at every step we want to:

     1. Calculate the Noise on the current image.
     2. Remove the Noise from the current image.
     3. Repeat/Iterate process 
    
* Final result is a **Less Noisy Image**


## More details of Stable Diffusion Purpose
* Unet is a very popular computer vision algorithm that is used to calculate the Noise in an image.
  * Unet is usually used to classify pixels in computer vision -- usually radiographic images in the medical domain. 
* It is important to know that we are often not able to estimate the error in 1 step, we do it slowly so the Diffusion process remains "Stable", hence the term "Stable Diffusion".
* If the process is performed too fast with BIG change this leads to significant instability in the results.
  * Why? the network is not able to learn the changes and patterns if this process is too quick and too big. 

![image](https://github.com/user-attachments/assets/7a7a4d64-4141-4e9e-8001-c5c6d5f78cb2)



# UNet
* UNet is a neural network algorithm that was initially utilized solely for medical image segmentation tasks.
* It is a type of CNN (convolutional neural network). 
* It was later repurposed for **image-2-image tasks** (e.g. given an input image, generate a new image)
* **In terms of Stable Diffusion, UNet is used to predict the Noise for each pixel in an image.**
* The image below demonstrates the UNet architecture:
   * The UNet we see below is simply described as having an encoder layer on the left, and a decoder on the right. This is what forms the "U" shape.
   * It has a Convolutional Encoder called "down sampling" and Decoder called "up sampling" --> which again is demonstrated in the "U" architecture. 

![image](https://github.com/user-attachments/assets/9548b76e-d069-4236-b73f-08bde84eb3e9)

## Why is UNet so popular?
* The algorithm learns modulations at **Multi-Scale** and **Multi-Abstraction** Levels.
* Lateral Skip Connections are the "Secret Sauce" compared to variational autoencoders or VAE. 
