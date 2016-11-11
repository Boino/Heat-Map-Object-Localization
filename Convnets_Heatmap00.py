
import numpy
from keras.optimizers import SGD
from convnetskeras.convnets import preprocess_image_batch, convnet
from convnetskeras.imagenet_tool import synset_to_dfs_ids
from convnetskeras.imagenet_tool import id_to_synset
import math
import Image
import cv2 as cv


import time
start_time = time.time()

path = 'convnets_keras/examples/'
image_path = 'tiger.jpg'


im = preprocess_image_batch([path + image_path],img_size=(256,256), crop_size=(224,224), color_mode="bgr")

#Create a List with all lables from ImageNet
List = open("ImageNet_Labels.txt").readlines()  


#Select only the Synset numbers from the List
List_synset = [item[:9] for item in List]

#Select only the Label from the List. Starts at col 12 so that number is not included
List_Labels = [item[13:] for item in List]

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
## For the VGG16, use this command
model = convnet('vgg_16',weights_path="weights/vgg16_weights.h5", heatmap=False)
## For the VGG19, use this one instead

# model = convnet('vgg_19',weights_path="weights/vgg19_weights.h5", heatmap=False)
model.compile(optimizer=sgd, loss='mse')

out = model.predict(im)

#Return the index of the maximum value of out
max_index = numpy.argmax(out)
min_index = numpy.argmin(out)
#Translate the max_index (e.g. 254) into the WordNet synset
wnid = id_to_synset(max_index)

ID = List_synset.index(wnid)
Label = List_Labels[ID]

print(Label)
print("--- %s seconds ---" % (time.time() - start_time))

# Although Redundant, we must run the model twice, one to find the Synset number (without the heatmap), and another one to calculate the multidimentional array that accounts for all the children of the class previously found.

im = preprocess_image_batch([path + image_path], color_mode="bgr")

model_heat = convnet('vgg_16',weights_path="weights/vgg16_weights.h5", heatmap=True)

model_heat.compile(optimizer=sgd, loss='mse')

out_heat = model_heat.predict(im)

#s = "n02084071"
#Gorilla_ID = 'n03000134'
#s_gorilla = 'n03000134'
ids = synset_to_dfs_ids(wnid)
heatmap = out_heat[0,ids].sum(axis=0)

# Then, we can get the image
import matplotlib.pyplot as plt

heat_path = 'heatmap_tiger.png'
plt.imsave(heat_path ,heatmap)



######################################################################

# Optimally, Image should be directly retreived from "heatmap", so that no external calls must be made

image_heat = cv.imread(heat_path,0)

#image_heat = cv.imread(heatmap, 0)
#image_heat = cv.imdecode(heatmap, 1)

heat_bin_row = heatmap.shape[0]
heat_bin_col = heatmap.shape[1]


heat_bin = numpy.zeros([heat_bin_row, heat_bin_col], dtype=numpy.uint8)
heat_bin.fill(255)
# Convert heat image to binary using Otsu's thresholding, to find the brightest area
(thres, heat_bin) = cv.threshold(image_heat, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

cv.imwrite("bin_" + heat_path, heat_bin) 

#coord = numpy.array([])
coord = []

# Choose mask to be an odd number, so that there can be a middle pixel.
mask = int(math.floor(0.2*(heat_bin_row+heat_bin_col)))
if mask % 2 == 0:
    mask = mask + 1  
    
pad_size = int(math.floor(mask/2))
#Resize image with border padding for the convolution

heat_bin_pad = numpy.lib.pad(heat_bin, ((pad_size, pad_size), (pad_size, pad_size)), 'minimum')

#######################################################################
count = 0

for i in range (1, heat_bin_row):
    for j in range (1, heat_bin_col):
        temp = heat_bin_pad[i-1:i+mask-1, j-1:j+mask-1]
        temp_sum = (sum(sum(temp)))/(mask)
        if (temp_sum > 240):
            #coord = numpy.append(coord, numpy.array([i, j]))
            coord.append([i, j])
            count = count + 1
            
            
            
            
coord_sum = numpy.sum(coord, 0)
coord_centre = [coord_sum[0]/len(coord), coord_sum[1]/len(coord)]

image_heat[coord_centre[0],coord_centre[1]] = 0
#plt.imsave("heatmap_sunglasses02.png",image_heat)

# Now we have the coordinates of the centre of our object in the heatmap image
# In order to find the coordinates in the original image, we have to do a coordinate mapping

image_row = im.shape[2]
image_col = im.shape[3]

map_factor_row = int(math.floor(image_row/heat_bin_row))
map_factor_col = int(math.floor(image_col/heat_bin_col))




#Choose one of the two below depending on the format of the image: RowsxCols or ColsxRows
coord_image = [coord_centre[0]*map_factor_row, coord_centre[1]*map_factor_col]
coord_image = [coord_centre[1]*map_factor_row, coord_centre[0]*map_factor_col]

# Jus to make sure the  prediction of the position is correct, I´ll print a red pixel on that possition in the originMexican_hairlessal image

im_test = Image.open(path + image_path)
pix = im_test.load()

""" #Prints all the points of coord in red in the original image
for i in range (0, len(coord)):
    pix[coord[i][0]*map_factor_row, coord[i][1]*map_factor_col] = (255, 0, 0)
""""    
    






# Prints a square of red pixels around the coord_image centre.

pix[coord_image[0]-1,coord_image[1]] = (255, 0, 0)
pix[coord_image[0]-1,coord_image[1]-1] = (255, 0, 0)
pix[coord_image[0],coord_image[1]-1] = (255, 0, 0)
pix[coord_image[0]+1,coord_image[1]-1] = (255, 0, 0)
pix[coord_image[0]+1,coord_image[1]] = (255, 0, 0)
pix[coord_image[0]+1,coord_image[1]+1] = (255, 0, 0)
pix[coord_image[0],coord_image[1]+1] = (255, 0, 0)
pix[coord_image[0]-1,coord_image[1]+1] = (255, 0, 0)
pix[coord_image[0],coord_image[1]] = (255, 0, 0)




"""
bin_test = Image.open("heat_bin.png")
pix_bin = bin_test.load()
pix_bin[coord_centre[0]-1,coord_centre[1]] = (255, 0, 0)
pix_bin[coord_centre[0]-1,coord_centre[1]-1] = (255, 0, 0)
pix_bin[coord_centre[0],coord_centre[1]-1] = (255, 0, 0)
pix_bin[coord_centre[0]-1,coord_centre[1]+1] = (255, 0, 0)
pix[coord_centre[0]+1,coord_centre[1]] = (255, 0, 0)
pix[coord_centre[0]+1,coord_centre[1]+1] = (255, 0, 0)
pix[coord_centre[0],coord_centre[1]] = (255, 0, 0)
pix[coord_centre[0],coord_centre[1]] = (255, 0, 0)
pix[coord_centre[0],coord_centre[1]] = (255, 0, 0)

"""




































"""







from keras.optimizers import SGD
from convnetskeras.convnets import preprocess_image_batch, convnet
from convnetskeras.imagenet_tool import synset_to_dfs_ids

im = preprocess_image_batch(['convnets_keras/examples/dog.jpg'], color_mode="bgr")

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model = convnet('alexnet',weights_path="weights/alexnet_weights.h5", heatmap=True)
model.compile(optimizer=sgd, loss='mse')

out = model.predict(im)

s = "n02084071"
ids = synset_to_dfs_ids(s)
heatmap = out[0,ids].sum(axis=0)

# Then, we can get the image
import matplotlib.pyplot as plt
plt.imsave("heatmap_dog.png",heatmap)
