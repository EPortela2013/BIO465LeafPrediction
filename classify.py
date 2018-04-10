import numpy as np
from PIL import Image

from image_processor import load_image
from image_processor import resize_image
import scipy.misc

image = load_image('images/cherry3.jpg')
resized_image = resize_image(image, 256, 256, 3, 'half_crop')
scipy.misc.toimage(resized_image, cmin=0.0, cmax=...).save('images/cherry4.png')

import caffe

# Uncomment the following lines if your caffe is set up to use GPU
#caffe.set_device(0)
#caffe.set_mode_gpu()

# Comment the following line if your caffe is set up to use GPU
caffe.set_mode_cpu()

net = caffe.Net('deploy.prototxt',
                'snapshots/snapshots__iter_54690.caffemodel',
                caffe.TEST)

#convert mean.binaryproto to numpy array
blob = caffe.proto.caffe_pb2.BlobProto()
data = open( 'mean_image/mean.binaryproto' , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )

# load input and configure preprocessing
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', arr[0].mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)


#load the image in the data layer
im = caffe.io.load_image('images/plantleaf2.png')
net.blobs['data'].data[...] = transformer.preprocess('data', im)
#compute
out = net.forward()

#print predicted labels
labels = np.loadtxt("labels.txt", str, delimiter='\t')
top_k = net.blobs['prob'].data[0].flatten().argsort()

plant_set = set()
index = 1

while len(plant_set) < 13:
    label = labels[top_k[-index]]
    divisor_index = label.index('___')
    plant = label[ : divisor_index]
    plant_set.add(plant)
    index += 1

print("\n\n\n****************************************\
            \nDo you know what type of plant this is?")
option_num = 1
sorted_plant_set = sorted(plant_set)
for x in sorted_plant_set:
    print("%d. %s" %(option_num, x))
    option_num += 1

print("%d. Don't know." % option_num )

try:
    chosen_option = int(input('>>> '))
except:
    chosen_option = option_num

if chosen_option > len(plant_set) or chosen_option <= 0:
    print("\nTop prediction is %s" % labels[top_k[-1]]);
else:
    plant = sorted_plant_set[chosen_option - 1]
    index = 1
    while True:
        label = labels[top_k[-index]]
        if label.find(plant) >= 0:
            print ("\nTop prediction for plant %s is %s" % (plant, label[len( plant + '___') : ]) )
            break
        else:
            index += 1
