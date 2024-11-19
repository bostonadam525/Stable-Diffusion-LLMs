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

