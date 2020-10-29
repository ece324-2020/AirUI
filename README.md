# AirUI

*Team Members: Aman Bhargava, Alice Zhou, Adam Carnaffan*

## Goal 

We aim to create an audio classification system for various gestures (scratches, taps, swipes, etc.) made on a wooden surface by a user. This will form the basis for a type of natural user interface known as ‘reality user interface’ (hence the name Artificially Intelligent Reality User Interface, or AirUI).

## Utility

This user interface method would offer an extremely low cost alternative for touch input in computer systems and could help to make technology more accessible, particularly for individuals who have difficulty using conventional user interfaces. 

## Rationale for Neural Network Model

Convolutional Neural Networks are a commonly used state-of-the-art method for audio classification, particularly when combined with time-frequency methods such as the short-time Fourier transform (STFT). 

## Overall Project Architecture

- The data will be pre-processed using audio production software Ableton Live (cropping), Scipy.io.wavfile (loading), Numpy (energy graph), and Scipy (spectrogram).
- Processed data will then be passed into one or multiple conventional ML models such as KNN, SVM, and/or Logistic Regression as a benchmark in the baseline stage. 
- MLP will be used in the development stage. 
- CNN (convolutional layers followed by fully connected layers) will be used in the final stage.

