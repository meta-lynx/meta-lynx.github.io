import tensorflow as tf
from keras_preprocessing import image # requires pillow
from sklearn.manifold import TSNE
import numpy as np
import os
import argparse
import configparser
import pandas as pd
import umap


NAMED_MODELS = ['MobileNet', 'MobileNetV2', 'ResNet50', 'ResNet101V2', 'ResNet152V2', 'ResNet50V2', \
    'VGG16', 'VGG19', 'InceptionV2', 'InceptionV3', 'DenseNet121', 'DenseNet169', 'DenseNet201', \
        'EfficientNetB0', 'EfficientNetB7', 'EfficientNetV2L', 'EfficientNetV2M', 'EfficientNetV2S']
IMG_TYPES=['jpg','jpeg','png']

def load_models(model_name, model_weights=None, number_classes=1000):

    # STANDARD MODEL ARCHITECTURE, IMAGENET OR USER-PROVIDED WEIGHTS
    if model_name == 'MobileNet':
        if model_weights:
            model_full = tf.keras.applications.mobilenet.MobileNet(weights=model_weights, include_top=True, classes=number_classes)
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-5].output)
        else:
            model_full = tf.keras.applications.mobilenet.MobileNet(weights='imagenet', include_top=True, classes=1000)
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-5].output)
    elif model_name == 'MobileNetV2':
        if model_weights:
            model_full = tf.keras.applications.mobilenet_v2.MobileNetV2(weights=model_weights, include_top=True, classes=number_classes)
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-2].output)
        else:
            model_full = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=True, classes=1000)
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-2].output)
    elif model_name == 'ResNet50':
        if model_weights:
            model_full = tf.keras.applications.resnet50.ResNet50(weights=model_weights, include_top=True, classes=number_classes)
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-2].output)
        else:
            model_full = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=True, classes=1000)
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-2].output)
    elif model_name == 'ResNet101V2':
        if model_weights:
            model_full = tf.keras.applications.resnet_v2.ResNet101V2(weights=model_weights, include_top=True, classes=number_classes)
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-2].output)
        else:
            model_full = tf.keras.applications.resnet_v2.ResNet101V2(weights='imagenet', include_top=True, classes=1000)
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-2].output)
    elif model_name == 'ResNet152V2':
        if model_weights:
            model_full = tf.keras.applications.resnet_v2.ResNet152V2(weights=model_weights, include_top=True, classes=number_classes)
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-2].output)
        else:
            model_full = tf.keras.applications.resnet_v2.ResNet152V2(weights='imagenet', include_top=True, classes=1000)
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-2].output)
    elif model_name == 'ResNet50V2':
        if model_weights:
            model_full = tf.keras.applications.resnet_v2.ResNet50V2(weights=model_weights, include_top=True, classes=number_classes)
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-2].output)
        else:
            model_full = tf.keras.applications.resnet_v2.ResNet50V2(weights='imagenet', include_top=True, classes=1000)
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-2].output)
    elif model_name == 'VGG16':
        if model_weights:
            model_full = tf.keras.applications.vgg16.VGG16(weights=model_weights, include_top=True, classes=number_classes)
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-2].output)
        else:
            model_full = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=True, classes=1000)
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-2].output)
    elif model_name == 'VGG19':
        if model_weights:
            model_full = tf.keras.applications.vgg19.VGG19(weights=model_weights, include_top=True, classes=number_classes)
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-2].output)
        else:
            model_full = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=True, classes=1000)  
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-2].output)
    elif model_name == 'InceptionV2':
        if model_weights:
            model_full = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(weights=model_weights, include_top=True, classes=number_classes)
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-2].output)
        else:
            model_full = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=True, classes=1000)  
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-2].output)
    elif model_name == 'InceptionV3':
        if model_weights:
            model_full = tf.keras.applications.inception_v3.InceptionV3(weights=model_weights, include_top=True, classes=number_classes)
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-2].output)
        else:
            model_full = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=True, classes=1000)  
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-2].output)
    elif model_name == 'DenseNet121':
        if model_weights:
            model_full = tf.keras.applications.densenet.DenseNet121(weights=model_weights, include_top=True, classes=number_classes)
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-2].output)
        else:
            model_full = tf.keras.applications.densenet.DenseNet121(weights='imagenet', include_top=True, classes=1000)  
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-2].output)
    elif model_name == 'DenseNet169':
        if model_weights:
            model_full = tf.keras.applications.densenet.DenseNet169(weights=model_weights, include_top=True, classes=number_classes)
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-2].output)
        else:
            model_full = tf.keras.applications.densenet.DenseNet169(weights='imagenet', include_top=True, classes=1000)  
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-2].output)
    elif model_name == 'DenseNet201':
        if model_weights:
            model_full = tf.keras.applications.densenet.DenseNet201(weights=model_weights, include_top=True, classes=number_classes)
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-2].output)
        else:
            model_full = tf.keras.applications.densenet.DenseNet201(weights='imagenet', include_top=True, classes=1000)  
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-2].output)
    elif model_name == 'EfficientNetB0':
        if model_weights:
            model_full = tf.keras.applications.efficientnet.EfficientNetB0(weights=model_weights, include_top=True, classes=number_classes)
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-3].output)
        else:
            model_full = tf.keras.applications.efficientnet.EfficientNetB0(weights='imagenet', include_top=True, classes=1000)  
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-3].output)
    elif model_name == 'EfficientNetB7':
        if model_weights:
            model_full = tf.keras.applications.efficientnet.EfficientNetB7(weights=model_weights, include_top=True, classes=number_classes)
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-3].output)
        else:
            model_full = tf.keras.applications.efficientnet.EfficientNetB7(weights='imagenet', include_top=True, classes=1000)  
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-3].output)
    elif model_name == 'EfficientNetV2L':
        if model_weights:
            model_full = tf.keras.applications.efficientnet_v2.EfficientNetV2L(weights=model_weights, include_top=True, classes=number_classes)
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-3].output)
        else:
            model_full = tf.keras.applications.efficientnet_v2.EfficientNetV2L(weights='imagenet', include_top=True, classes=1000)  
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-3].output)
    elif model_name == 'EfficientNetV2L':
        if model_weights:
            model_full = tf.keras.applications.efficientnet_v2.EfficientNetV2M(weights=model_weights, include_top=True, classes=number_classes)
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-3].output)
        else:
            model_full = tf.keras.applications.efficientnet_v2.EfficientNetV2M(weights='imagenet', include_top=True, classes=1000)  
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-3].output)
    elif model_name == 'EfficientNetV2M':
        if model_weights:
            model_full = tf.keras.applications.efficientnet_v2.EfficientNetV2S(weights=model_weights, include_top=True, classes=number_classes)
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-3].output)
        else:
            model_full = tf.keras.applications.efficientnet_v2.EfficientNetV2S(weights='imagenet', include_top=True, classes=1000)  
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-3].output)
    elif model_name == 'Xception':
        if model_weights:
            model_full = tf.keras.applications.xception.Xception(weights=model_weights, include_top=True, classes=number_classes)
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-2].output)
        else:
            model_full = tf.keras.applications.xception.Xception(weights='imagenet', include_top=True, classes=1000)  
            model_feat = tf.keras.Model(inputs = model_full.input, outputs=model_full.layers[-2].output)
    else:
        raise ValueError('This script runs with standard model architecures only')
    # SEPARATE FILE FOR USING OWN MODELS WHERE RECOMMENDED LAYERS ARE GIVEN FOR EACH MODEL

    return model_feat, model_full


def preprocess_image(model_name, image_in):

    # STANDARD MODEL ARCHITECTURE, IMAGENET OR USER-PROVIDED WEIGHTS
    if model_name == 'MobileNet':
        image_out = tf.keras.applications.mobilenet.preprocess_input(image_in)
    elif model_name == 'MobileNetV2':
        image_out = tf.keras.applications.mobilenet_v2.preprocess_input(image_in)
    elif model_name == 'ResNet50':
        image_out = tf.keras.applications.resnet50.preprocess_input(image_in)
    elif model_name == 'ResNet101V2' or model_name == 'ResNet152V2' or model_name == 'ResNet50V2':
        image_out = tf.keras.applications.resnet_v2.preprocess_input(image_in)
    elif model_name == 'VGG16':
        image_out = tf.keras.applications.vgg16.preprocess_input(image_in) 
    elif model_name == 'VGG19':
        image_out = tf.keras.applications.vgg19.preprocess_input(image_in)
    elif model_name == 'InceptionV2':
        image_out = tf.keras.applications.inception_resnet_v2.preprocess_input(image_in)
    elif model_name == 'InceptionV3':
        image_out = tf.keras.applications.inception_v3.preprocess_input(image_in)
    elif model_name == 'DenseNet121' or model_name == 'DenseNet169' or model_name == 'DenseNet201':
        image_out = tf.keras.applications.densenet.preprocess_input(image_in)
    elif model_name == 'EfficientNetB0' or model_name == 'EfficientNetB7':
        image_out = tf.keras.applications.efficientnet.preprocess_input(image_in)
    elif model_name == 'EfficientNetV2L' or model_name == 'EfficientNetV2M' or model_name == 'EfficientNetV2S':
        image_out = tf.keras.applications.efficientnet_v2.preprocess_input(image_in)
    else:
        raise ValueError('This script runs with standard model architecures')
        
    return image_out


def create_data_series(input_list, index_list, series_name):

    # create a pandas data series
    new_series = pd.Series(input_list, index=index_list, name=series_name)

    return new_series


def dimensionality_reduction(features, n_dimensions=3,  method='TSNE', perplexity=50, n_iterations=500):

    if method=='TSNE':
        tsne = TSNE(n_components=n_dimensions, verbose=1, perplexity=perplexity, n_iter=n_iterations)
        reduced_coords = tsne.fit_transform(features)
    elif method=='UMAP':
        ufit = umap.UMAP(n_components=n_dimensions)
        reduced_coords = ufit.fit_transform(features)
    else:
        print("Unrecognised dimensionality reduction method")

    return reduced_coords


def extract_model_metadata(model, model_name, image_list, image_shape=(224,224)):

    # return max confidence and prediction using full model
    conf_list = []
    pred_list = []
    for ii in image_list:
        test_image = str(ii)
        img = image.load_img(test_image, target_size=image_shape)
        img_arr = image.img_to_array(img)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = preprocess_image(model_name, img_arr)

        all_predictions = model.predict(img_arr)[0]    
        confidence = np.max(all_predictions)
        prediction = np.argmax(all_predictions)
        conf_list.append(confidence)
        pred_list.append(prediction)

    Confidence = create_data_series(conf_list, image_list, 'Confidence')
    Predictions = create_data_series(pred_list, image_list, 'Prediction')

    return (Confidence, Predictions)


def extract_model_features(model, model_name, image_list, image_shape=(224,224), flatten=True, create_series=True):

    features_list = []
    for ii in image_list:
        test_image = str(ii)
        img = image.load_img(test_image, target_size=image_shape)
        img_arr = image.img_to_array(img)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = preprocess_image(model_name, img_arr)

        features = model.predict(img_arr)[0]
        if flatten:
            features_list.append(features.flatten())
        else:
            features_list.append(features)

    if create_series == True:
        features_series = create_data_series(features_list, image_list, 'Features')
    else:
        features_series = features_list

    return features_series


def extract_embedding_coordinates(model, model_name, image_list, image_shape=(224,224), method='TSNE', stem='tsne_coords_'):

    features_list = extract_model_features(model, model_name, image_list, image_shape, flatten=True, create_series=False)

    coords = dimensionality_reduction(features_list, 3, method)

    # Only allow for 3 hard coded for now
    feature_len = len(coords[0,:])
    if feature_len == 3:
        XCoord = create_data_series(coords[:,0], image_list, stem+'0')
        YCoord = create_data_series(coords[:,1], image_list, stem+'1')
        ZCoord = create_data_series(coords[:,2], image_list, stem+'2')
        all_coords = (XCoord, YCoord, YCoord)
    else:
        all_coords = create_data_series([0]*len(image_list), image_list)
    
    return all_coords


def main():

    parser = argparse.ArgumentParser(description='Create CSV file for Metascatter')
    parser.add_argument("config_file", type=str, help="Path to config file")
    args = parser.parse_args()

    # Read in configuration file
    config = configparser.ConfigParser()
    config.read(args.config_file)

    # Read in model variables
    model_name = config['MODEL VARIABLES']['model_name']
    model_weights = config['MODEL VARIABLES']['model_weights']
    image_size = int(config['MODEL VARIABLES']['image_size'])

    # Read in data folders
    # Labelled data
    labelled_folders = config['LABELLED IMAGE FOLDERS']['labelled_folder_list'][1:-1].split(' ')
    labelled_sources = config['LABELLED IMAGE FOLDERS']['labelled_folder_sources'][1:-1].split(' ')
    
    # Unlabelled data
    unlabelled_folders = config['UNLABELLED IMAGE FOLDERS']['unlabelled_folder_list'][1:-1].split(' ')
    unlabelled_sources = config['UNLABELLED IMAGE FOLDERS']['unlabelled_folder_sources'][1:-1].split(' ')

    # Output file
    output_filename = config['OUTPUT FILENAME']['savefile']

    # create image, ground truth label and source list 
    image_list=[]
    class_list=[]
    source_list=[]
    # Loop over labelled folders
    for ff in range(len(labelled_folders)):
        for image_path in os.listdir(labelled_folders[ff]):
            classfolder = os.path.join(labelled_folders[ff],image_path)
            for image in os.listdir(classfolder):
                image_list.append(os.path.join(image_path,classfolder,image))
                class_list.append(image_path)
                source_list.append(labelled_sources[ff])

    number_classes = len(set(class_list))

    # Loop over unlabelled folders
    for ff in range(len(unlabelled_folders)):
        for image in os.listdir(unlabelled_folders[ff]):
            image_list.append(os.path.join(unlabelled_folders[ff],image))
            class_list.append('Unlabelled')
            source_list.append(unlabelled_sources[ff])

    Labels = create_data_series(class_list, image_list, 'Labels')
    Sources = create_data_series(source_list, image_list, 'Sources')

    # Get model(s)
    feature_model, full_model = load_models(model_name, model_weights, number_classes) 

    # Extract embedding coordinates and metadata
    # if model exists
    coords_t = extract_embedding_coordinates(feature_model, model_name, image_list, (image_size, image_size), 'TSNE', 'tsne_coords_') # TSNE Coordinates
    #coords_u = extract_embedding_coordinates(feature_model, model_name, image_list, (image_size, image_size), 'UMAP', 'umap_coords_') # UMAP Coordinates
    meta_data = extract_model_metadata(full_model, model_name, image_list, (image_size, image_size)) # Prediction, Confidence
    ## TO ADD - if no model, use image intensities for features

    # create pandas dataframe from lists
    all_data = pd.concat([Sources, Labels], axis=1)
    all_data = pd.concat([all_data, coords_t[0], coords_t[1], coords_t[2]], axis=1)
    #all_data = pd.concat([all_data, coords_u[0], coords_u[1], coords_u[2]], axis=1)
    all_data = pd.concat([all_data, meta_data[0], meta_data[1]], axis=1)

    # save dataframe as csv   
    # remove common prefix
    common_length = len(os.path.commonprefix(image_list))
    relative_path_list = ['/'+item[common_length:] for item in image_list]
    all_data.index=relative_path_list
    all_data.index.names=["Image_path"]
    all_data.to_csv(output_filename)


if __name__ == "__main__":
    main()
