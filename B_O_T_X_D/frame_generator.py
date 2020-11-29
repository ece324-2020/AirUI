# Imports
import numpy as np
import spacy as spc
import pandas as pd
import math as mt
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torchtext as tt
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.io import wavfile
import scipy.io

import librosa
import librosa.display
import IPython.display as ipd

from tqdm import tqdm


numOfConvLayers = 4
numOfKernels = 12
numOfNeurons = 12
learningRate = 0.1
batchSize = 48
numOfEpochs = 200
kernelSize = 3
activation = F.relu       
optimizer_func = torch.optim.SGD
loss_fnc = nn.CrossEntropyLoss()
batchNorm = False
numOfFCLayers = 2
testSize = 0.2
valSize = 0.2
seed = 42

# Model Init
class CNN(nn.Module):
  def __init__(self, input_dims, numOfKernels, numOfNeurons, kernelSize, numOfConvLayers, batchNorm):
    super(CNN, self).__init__()         
    self.numOfKernels = numOfKernels
    self.batchNorm = batchNorm
    self.numOfConvLayers = numOfConvLayers

    # Convolutional Layers
    self.conv1 = nn.Conv2d(3,numOfKernels, kernelSize)
    self.conv2 = nn.Conv2d(numOfKernels,numOfKernels, kernelSize)
    self.conv_BN = nn.BatchNorm2d(numOfKernels)

    # Determine the output size after the convolutional layer
    fullLayerSize_x = input_dims[1]
    fullLayerSize_y = input_dims[0]
    for i in range (self.numOfConvLayers):
      fullLayerSize_x = (fullLayerSize_x-kernelSize+1)//2
      fullLayerSize_y = (fullLayerSize_y-kernelSize+1)//2

    # Error check the output size
    if fullLayerSize_x <= 0 or fullLayerSize_y <= 0:
      raise Exception("Too many convolutional layer for the input size, please decrease numOfConvLayers.")

    # Fully connected layers
    self.fc1 = nn.Linear(numOfKernels*fullLayerSize_x*fullLayerSize_y, numOfNeurons)
    self.fc1_BN = nn.BatchNorm1d(numOfNeurons)
    self.pool = nn.MaxPool2d(2,2)
    self.fc2 = nn.Linear(numOfNeurons, 6)
    self.fc2_BN = nn.BatchNorm1d(6)

  def forward(self, x):
    activation = F.relu   

    if self.batchNorm == True:
      x = self.pool(activation(self.conv_BN(self.conv1(x))))
      for i in range (self.numOfConvLayers - 1):
        x = self.pool(activation(self.conv_BN(self.conv2(x))))
      x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
      x = activation(self.fc1_BN(self.fc1(x)))
      x = self.fc2_BN(self.fc2(x))
    else: 
      x = self.pool(activation(self.conv1(x)))
      for i in range (self.numOfConvLayers - 1):
        x = self.pool(activation(self.conv2(x)))
      x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
      x = activation(self.fc1(x))
      x = self.fc2(x)
    return x

# Get model
model = torch.load('mdl.pt')
model.eval()
trans = transforms.Compose([transforms.CenterCrop((100, 65)), transforms.ToTensor(), transforms.Normalize([0.4275, 0.4275, 0.4275], [0.2293, 0.2293, 0.2293])])
# trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.4275, 0.4275, 0.4275], [0.2293, 0.2293, 0.2293])])


def snufflupugus(spec_in, show_model_input = False):
    im = Image.fromarray(spec_in.copy())
    clean_scan = trans(im)

    # print('FROM SNUFFLUPUGUS: Shape of im is {}'.format(im.shape))
    model_input = clean_scan.reshape(1, 3, 100, 65)
    predict = model(clean_scan.reshape(1, 3, 100, 65))

    if show_model_input:

        sli = model_input[0,:,:,:]
        guh = sli.numpy()
        print('Max of input is: {}'.format(np.max(guh)))


        # plt.imshow(guh[2,:,:], cmap='gray')
        # plt.title('Slice 2 of model input!')
        # plt.show()

        # plt.imshow(guh[1,:,:], cmap='gray')
        # plt.title('Slice 1 of model input!')
        # plt.show()

        # plt.imshow(guh[0,:,:], cmap='gray')
        # plt.title('Slice 0 of model input!')
        # plt.show()



    return predict


def evaluate(model, data_loader, batchNorm):   # used on validation and test set
  total_corr = 0
  totalLoss = 0
  if batchNorm:
    model.eval()
  else:
    model.train()
  for i, (inputs, labels) in enumerate(data_loader):
    oneh_labels = oneh_classes[labels]
    outputs = model(inputs)
    if str(loss_fnc) == "CrossEntropyLoss()":                # CrossEntropyLoss does not accept one-hot labels
      loss = loss_fnc(input=outputs, target=labels) 
    else: 
      loss = loss_fnc(input=outputs, target=oneh_labels) 
    total_corr += numOfCorrectPredictions(outputs, labels) 
    totalLoss += loss.item()
  accuracy = float(total_corr)/len(data_loader.dataset)
  loss = totalLoss/len(data_loader)
  return (loss, accuracy)

def get_dataloader():
    resizeMethod = "scale"
    # Custom Horizontal Pad Pipeline 
    class SquarePad:
      def __call__(self, image):
        w = image.size[0]  
        maxw = 130            #pad to 130
        hp = (maxw - w) // 2
        remainder = (maxw - w) % 2
        padding = (hp, 0, hp+remainder, 0)
        return transforms.functional.pad(image, padding, 0, 'constant')
    # Input overfit, train, val, test dataset 
    # Load un-normalized data to compute mean and standard deviation.
    image_size = (100, 130) # size of the input image we want to pass into the model (100 height is standard already)
                           # In our dataset, (100, 65) is size of the smallest image,
                           # (100, 130) is size of the biggest image.
    ResizeMethods = {"pad": [SquarePad()], 
                     "crop": [transforms.CenterCrop(image_size)], 
                     "scale": [transforms.Resize(image_size)]
                     }


    # padding/cropping pipeline for unnormalized dataset
    data_pipe = transforms.Compose(ResizeMethods[resizeMethod] +
        [transforms.ToTensor()])

    path = '../Full_Dataset/Spectrograms'

    dataset_raw = torchvision.datasets.ImageFolder(\
                root=path, \
                transform=data_pipe)
    loader = DataLoader(dataset_raw, batch_size=len(dataset_raw))
    data = next(iter(loader))
    mean = torch.mean(data[0], (0,2,3))
    std = torch.std(data[0], (0,2,3))

    # padding/cropping pipeline for normalized datasets
    data_pipeline_normal_reg = transforms.Compose(ResizeMethods[resizeMethod] + 
        [transforms.ToTensor(), transforms.Normalize(mean, std)])
    # data_pipeline_normal_over = transforms.Compose(ResizeMethods[resizeMethod] + 
        # [transforms.ToTensor(), transforms.Normalize(mean_overfit, std_overfit)])

    # Normalized dataset
    dataset = torchvision.datasets.ImageFolder(\
                root=path, \
                transform=data_pipeline_normal_reg)

    # create train, val, test, and overfit dataset
    train_set, test_set = train_test_split(dataset, test_size=testSize, random_state=seed)   # split test
    train_set, val_set = train_test_split(train_set, test_size=valSize/(1-testSize), random_state=seed)   # split train to train and val

    train_loader = DataLoader(train_set, batch_size=batchSize, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batchSize, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batchSize, shuffle=True)
    # overfit_loader = DataLoader(overfit_set, batch_size=batchSize, shuffle=True)

    dataset_raw = torchvision.datasets.ImageFolder(\
            root=path, \
            transform=data_pipe)
    loader = DataLoader(dataset_raw, batch_size=len(dataset_raw))

    return loader

def generate_spectrogram(file_path):
    FS_global = 44100 # Global value for sampling rate (set in DAW)

    num_mels = 100


    FS, audio = wavfile.read(file_path)

    wave = (audio[:,0] + audio[:,1])*0.5

    print('Waveform shape: {}'.format(wave.shape))
    plt.figure(figsize = (10, 3))
    plt.title('Audio Waveform')
    plt.plot(wave)
    plt.savefig('test_outputs/waveform.png', dpi=300)
    plt.show()

    filter_banks = librosa.filters.mel(n_fft=2048, sr=FS, n_mels=num_mels)
    mel_spectrogram = librosa.feature.melspectrogram(wave, sr=FS, n_fft=2048, hop_length=512, n_mels=num_mels)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    plt.imsave('test_outputs/test_spectrogram.png', log_mel_spectrogram, cmap='gray')

"""
=== FRAME GENERATOR ===

This script generates 'frames' for animations showing how the model performs on 
live incoming data. 'Frames' include png-formatted slices of spectrograms along
with bar graphs showing the probabilities of each class. There is one frame per
pixel along the input audio. Spectrograms are analyzed in windows. 
"""

def plot_class_probabilities(raw_output, names, savename):
    probabilities = softmax(raw_output)
    predicted_class_num = np.argmax(probabilities)

    plt.figure(figsize = (10, 5))
    plt.bar(names, np.asarray(probabilities)*100)
    ax = plt.gca()
    ax.set_ylim([0,100])
    plt.title('Probabilities of Each Class')
    plt.savefig(savename, dpi=150)
    plt.close()

def curr_prediction_output(raw_output, names, savename):
    probabilities = softmax(raw_output)
    predicted_class_num = np.argmax(probabilities)
    class_name = names[predicted_class_num]

    img = Image.new('RGB', (700, 100), color = (255,255,255))
 
    fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 40)
    d = ImageDraw.Draw(img)
    d.text((30, 20), "Prediction: {}".format(class_name), font=fnt, fill=(50, 50, 50))
     
    img.save(savename)
    plt.close()




def generate_frames():
    """
    4. Record some sample audio.
        - Talking
        - ['Circle Scratch', 
        - 'Fingernail Tap', 
        - 'Fingertip Tap', 
        - 'Silence', 
        - 'Vertical Scratch', 
        - 'W Scratch']
    5. Convert into L O N G spectrogram. 
    6. Create pixel-by-pixel selector loop (65 pixels long, 100 pixels tall)
    7. Repeatedly run the algorithm, exporting the following each time
    """
    names = ['Circle Scratch', 'Fingernail Tap', 'Fingertip Tap', 'Silence', 'Vertical Scratch', 'W Scratch']

    # Generating and importing spectrogram image.
    generate_spectrogram('long_spec_wav.wav')
  
    # Importing and showing image
    image = cv2.imread('test_outputs/test_spectrogram.png', 1) 
    width = image.shape[1]


    skip = 1
    n = 1 # Number of items in rolling average
    roll_list = [[0,0,0,0,0,0]]

    for i in tqdm(range(0, width-65, skip)):
        im = image[:,i:i+65,:]
        # plt.imshow(im)
        # plt.show()

        # Running model
        out = snufflupugus(im).detach()[0]
        out = list(out)


        for j in range(len(out)):
            out[j] = float(out[j])

        roll_list = [out]+roll_list

        if len(roll_list) > n:
            roll_list.pop()

        roll_np = np.asarray(roll_list)

        cur_avg = np.sum(roll_np, 0)/roll_np.shape[0]

        plot_class_probabilities(cur_avg, names, 'test_outputs/probabilities/{}.png'.format(i//skip))
        curr_prediction_output(cur_avg, names, 'test_outputs/verdict/{}.png'.format(i//skip))


        plt.imsave('test_outputs/spec_snapshot/{}.png'.format(i//skip), im, cmap='gray')

def test_frame_visualize(file_path, output_name, gnd_trth='Unspeciffied'):

    names = ['Circle Scratch', 'Fingernail Tap', 'Fingertip Tap', 'Silence', 'Vertical Scratch', 'W Scratch']

    # Get image
    image = cv2.imread(file_path, 1) 
    width = image.shape[1]

    # Convert to proper type and size
    if width > 65:
        pad = (width-65)//2
        im = image[:,pad:pad+65,:]
    else:
        im = image

    # Run through model
    out = snufflupugus(im).detach()[0]
    out = list(out)
    for j in range(len(out)):
        out[j] = float(out[j])

    # Compute probabilities
    # probabilities = softmax(out)
    probabilities = out
    predicted_class_num = np.argmax(probabilities)

    fig3 = plt.figure(constrained_layout=True, figsize=(15, 6))
    gs = fig3.add_gridspec(1, 3)

    f3_ax1 = fig3.add_subplot(gs[0, 0:2])
    f3_ax1.set_title('Class Probabilities')
    f3_ax1.bar(names, np.asarray(probabilities)*100)

    f3_ax2 = fig3.add_subplot(gs[0, 2])
    f3_ax2.set_title('Spectrogram')
    f3_ax2.imshow(im)
    plt.title('Source: {}; Ground Truth: {}'.format(file_path, gnd_trth))
    plt.savefig(output_name, dpi=150)
    plt.close()

def generate_examples():
    out_file = 'sanity_check/'
    in_dir = '../Full_Dataset/Spectrograms/'

    types = ['Circle_Scratches', 'Fingernail_Taps', 'Fingertip_Taps', 'Silences', 'Vertical_Scratches', 'W_Scratches']
    
    for type_ in types:
        for num in range(60):
            ind = num*20
            test_frame_visualize(in_dir+type_+'/'+str(ind)+'.png', out_file+type_+str(ind)+'.png', gnd_trth=type_)

def check_acc(in_dir, highest_num, expected_class_ind):
    num_total = 0
    num_correct = 0

    types = ['Circle_Scratches', 'Fingernail_Taps', 'Fingertip_Taps', 'Silences', 'Vertical_Scratches', 'W_Scratches']

    print('Starting check of {}...'.format(types[expected_class_ind]))

    for i in tqdm(range(highest_num)):
        pth = in_dir+'/'+str(i)+'.png'

        num_total += 1

        # Get image
        image = cv2.imread(pth, -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        # image = Image.open(pth)
        # width = image.shape[1]

        # # Convert to proper type and size
        # if width > 65:
        #     pad = (width-65)//2
        #     im = image[:,pad:pad+65,:]
        # else:
        #     im = image

        # Run through model
        out = snufflupugus(image).detach()[0]
        
        out = list(out)
        for j in range(len(out)):
            out[j] = float(out[j])

        # Compute probabilities
        # probabilities = softmax(out)
        probabilities = out
        predicted_class_num = np.argmax(out)



        if predicted_class_num == expected_class_ind:
            num_correct+=1

    acc = num_correct/num_total

    return acc

def bot_acc_check():
    in_dir = '../Full_Dataset/Spectrograms/'
    types = ['Circle_Scratches', 'Fingernail_Taps', 'Fingertip_Taps', 'Silences', 'Vertical_Scratches', 'W_Scratches']
    
    out_file = 'sanity_check/'

    results = []

    for ind, type_ in enumerate(types):
        results.append(check_acc(in_dir+type_, 1440, ind))
        test_frame_visualize(in_dir+type_+'/'+str(666)+'.png', out_file+type_+str(666)+'.png', gnd_trth=type_)


    for i in range(len(results)):
        print("Accuracy on class {}: {}".format(types[i], results[i]))
        
def main():
    bot_acc_check()
main()