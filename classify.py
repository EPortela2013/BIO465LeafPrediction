import numpy as np
from PIL import Image
import optparse
from image_processor import load_image
from image_processor import resize_image
import scipy.misc
import os
import threading

os.environ['GLOG_minloglevel'] = '3' # Errors only
import caffe

class Predictor(object):

    def __init__(self):
        # Uncomment the following lines if your caffe is set up to use GPU
        # caffe.set_device(0)
        # caffe.set_mode_gpu()

        # Comment the following line if your caffe is set up to use GPU
        caffe.set_mode_cpu()

        # Define image settings
        self.image_width = 256
        self.image_height = 256
        self.image_channels = 3

        # Define caffe settings for net
        self.prototxt_file = 'deploy.prototxt'
        self.caffe_model = 'snapshots/snapshots__iter_54690.caffemodel'
        self.caffe_mode = caffe.TEST
        self.mean_binaryproto = 'mean_image/mean.binaryproto'

        self.num_plants = 14

        # convert mean.binaryproto to numpy array
        blob = caffe.proto.caffe_pb2.BlobProto()
        data = open(self.mean_binaryproto, 'rb').read()
        blob.ParseFromString(data)
        self.arr = np.array(caffe.io.blobproto_to_array(blob))

        self.image_name = None
        self.labels_file_name = 'labels.txt'
        self.labels = None
        self.sorted_plant_set = None
        self.images_dir = 'images'
        self.results_file_name = 'results.tsv'
        self.rotation_degrees = 0.0

        self.parse_options()

        self.results_file = open(self.results_file_name, "w")
        self.write_result_headers()

        self.labels = np.loadtxt(self.labels_file_name, str, delimiter="\t")

    @staticmethod
    def split_label(label):
        divisor_index = label.index('___')
        return label[: divisor_index], label[divisor_index + len('___'):]

    @staticmethod
    def delete_image(image_name):
        os.remove(image_name)

    def write_result_headers(self):
        self.results_file.write("Label\tStraight Prediction\tPlant Name Given\tNumber of Images\n")

    def write_result(self, label, straight_prediction, plant_name_given, num_images):
        split_label = self.split_label(label)
        self.results_file.write("%s - %s\t%f\t%f\t%d\n" %
                                (split_label[0], split_label[1], straight_prediction, plant_name_given, num_images))


    def predict(self, image_name=None, expected_label=None):

        if not image_name:
            image_name = self.image_name

        if not expected_label:
            interactive = True
        else:
            interactive = False
            expected_plant = self.split_label(expected_label)[0]

        net = caffe.Net(self.prototxt_file, self.caffe_model, self.caffe_mode)

        # load input and configure preprocessing
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_mean('data', self.arr[0].mean(1).mean(1))
        transformer.set_transpose('data', (2,0,1))
        transformer.set_channel_swap('data', (2,1,0))
        transformer.set_raw_scale('data', 255.0)


        #load the image in the data layer
        im = caffe.io.load_image(image_name)
        net.blobs['data'].data[...] = transformer.preprocess('data', im)

        #compute
        net.forward()

        top_k = net.blobs['prob'].data[0].flatten().argsort()
        if interactive:
            probabilities = sorted(net.blobs['prob'].data[0].flatten())
        plant_set = set()
        index = 1

        while len(plant_set) < self.num_plants:
            label = self.labels[top_k[-index]]
            plant = self.split_label(label)[0]
            plant_set.add(plant)
            index += 1

        interactive and print("****************************************\
                    \nDo you know what type of plant this is?")
        option_num = 1
        sorted_plant_set = sorted(plant_set)
        for x in sorted_plant_set:
            interactive and print("%d. %s" %(option_num, x))
            option_num += 1

        interactive and print("%d. Don't know." % option_num )

        if interactive:
            try:
                chosen_option = int(input('>>> '))
            except:
                chosen_option = option_num
        else:
            chosen_option = sorted_plant_set.index(expected_plant) + 1

        if interactive and (chosen_option > len(plant_set) or chosen_option <= 0):
            print("\nTop prediction is %s with probability %f" % (self.labels[top_k[-1]], probabilities[-1]))
            return True, True
        else:
            plant = sorted_plant_set[chosen_option - 1]
            index = 1
            while True:
                label = self.labels[top_k[-index]]
                if label.find(plant) >= 0:
                    interactive and print("\nTop prediction for plant %s is %s with probability %f" % (plant, label[len( plant + '___') : ], probabilities[-index]) )
                    return expected_label == self.labels[top_k[-1]], expected_label == label
                else:
                    index += 1

    def predict_images(self):
        if self.image_name:
            image_name = self.resize_if_needed(self.image_name)
            self.predict(image_name)
            if self.image_name != image_name:
                self.delete_image(image_name)
            return

        # Predict all images in images folder

        for path in os.listdir(self.images_dir):
            if os.path.isdir(self.images_dir + '/' + path):
                label_path = self.images_dir + '/' + path
                self.predict_images_dir(label_path)


    def predict_images_dir(self, path):
        num_images = 0
        num_correct_straight = 0
        num_correct_plant_given = 0
        last_dir_index = path.find('/')
        expected_label = path[last_dir_index + 1:]

        print("Making predictions for label %s" % expected_label)

        for possible_image in os.listdir(path):
            if not os.path.isdir(path + '/' + possible_image):
                image_name = path + '/' + possible_image
                new_image_name = self.resize_if_needed(image_name)

                if self.rotation_degrees != 0.0:
                    # Rotate image as well
                    image = Image.open(new_image_name)
                    image.rotate(self.rotation_degrees)
                    temp_name = new_image_name + '-rotated.png'
                    if image_name != new_image_name:
                        self.delete_image(new_image_name)
                    image.save(temp_name)
                    new_image_name = temp_name

                result = self.predict(new_image_name, expected_label)
                num_images += 1
                if result[0]:
                    num_correct_straight += 1
                if result[1]:
                    num_correct_plant_given += 1

                if image_name != new_image_name:
                    self.delete_image(new_image_name)

        if num_images > 0:
            ratio_straight = num_correct_straight / num_images
            ratio_plant_name_given = num_correct_plant_given / num_images
            self.write_result(expected_label, ratio_straight, ratio_plant_name_given, num_images)



    def resize_if_needed(self, image_name):
        image = load_image(image_name)
        width, height = image.size

        if width == self.image_width and height == self.image_height:
            return image_name

        new_image_name = image_name + '-resized.png'
        resized_image = resize_image(image, self.image_width, self.image_height,
                                        self.image_channels, 'half_crop')
        scipy.misc.toimage(resized_image, cmin=0.0, cmax=...).save(new_image_name)

        return new_image_name


    def parse_options(self):
        parser = optparse.OptionParser(usage="%prog [options]",
                                       version="%prog 0.1")

        parser.add_option("-i", "--image", type="str", dest="image",
                          default=None,
                          help="Path of image on which to make prediction")

        parser.add_option("-l", "--labels", type="str", dest="labels",
                          default=self.labels_file_name,
                          help="Path of labels file")

        parser.add_option("-s", "--save-results", type="str", dest="results_file_name",
                          default=self.results_file_name,
                          help="Path of where to write results")

        parser.add_option("-r", "--rotation-degrees", type="float", dest="rotation_degrees",
                          default=self.rotation_degrees,
                          help="Path of where to write results")

        (options, args) = parser.parse_args()
        self.image_name = options.image
        self.labels_file_name = options.labels
        self.results_file_name = options.results_file_name
        self.rotation_degrees = options.rotation_degrees


if __name__ == '__main__':
    predictor = Predictor()
    predictor.predict_images()
