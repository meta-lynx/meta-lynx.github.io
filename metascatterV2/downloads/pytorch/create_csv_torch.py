import torch
import torchvision.models as models
import os
import argparse
import configparser
import pandas as pd
import copy
from sklearn.manifold import TSNE
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import transforms_config
import umap
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

NAMED_MODELS = ['AlexNet', 'MobileNetV2', 'MobileNetV3L', 'MobileNetV3S', 'ResNet18',  \
    'VGG16', 'InceptionV3', 'DenseNet161', 'GoogleNet', 'SqueezeNet' ]
IMG_TYPES=['jpg','jpeg','png']

# PyTorch model functions
def load_models(model_name, model_weights=None, number_classes=1000):

    if model_name == 'AlexNet':
        model_full = models.alexnet(pretrained=False)
        model_full.classifier[6]=torch.nn.Linear(4096,number_classes)
        model_full.load_state_dict(torch.load(model_weights))
        model_feat = copy.deepcopy(model_full)
        features = list(model_full.classifier.children())[:-1]
        model_feat.classifier=torch.nn.Sequential(*features)
    elif model_name == 'ResNet18':
        model_full = models.resnet18(pretrained=False)
        model_full.fc = torch.nn.Linear(512, number_classes)
        model_full.load_state_dict(torch.load(model_weights))
        model_feat = copy.deepcopy(model_full)
        model_feat.fc = torch.nn.Identity()
    elif model_name == 'VGG16':
        model_full = models.vgg16(pretrained=False)
        model_full.classifier[6] = torch.nn.Linear(4096,number_classes)
        model_full.load_state_dict(torch.load(model_weights))
        model_feat = copy.deepcopy(model_full)
        features = list(model_full.classifier.children())[:-1]
        model_feat.classifier=torch.nn.Sequential(*features)
    elif model_name == 'SqueezeNet':
        model_full = models.squeezenet1_0(pretrained=False)
        model_full.classifier[1] = torch.nn.Conv2d(512, number_classes, kernel_size=(1,1), stride=(1,1)) 
        model_full.load_state_dict(torch.load(model_weights))
        model_feat = copy.deepcopy(model_full)
        features = list(model_full.classifier.children())[:-3]
        model_feat.classifier=torch.nn.Sequential(*features)
    elif model_name == 'DenseNet161':
        model_full = models.densenet161(pretrained=False)
        model_full.classifier = torch.nn.Linear(2208, number_classes)
        model_full.load_state_dict(torch.load(model_weights))
        model_feat = copy.deepcopy(model_full)
        model_feat.classifier=torch.nn.Identity()
    elif model_name == 'DenseNet121':
        model_full = models.densenet121(pretrained=False)
        model_full.classifier = torch.nn.Linear(1024, number_classes)
        model_full.load_state_dict(torch.load(model_weights))
        model_feat = copy.deepcopy(model_full)
        model_feat.classifier=torch.nn.Identity()
    elif model_name == 'InceptionV3':
        model_full = models.inception_v3(pretrained=False)
        model_full.AuxLogits.fc = torch.nn.Linear(768, number_classes)
        model_full.fc = torch.nn.Linear(2048, number_classes)
        model_full.load_state_dict(torch.load(model_weights))
        model_feat = copy.deepcopy(model_full)
        model_feat.fc =torch.nn.Identity()
    elif model_name == 'GoogleNet':
        model_full = models.googlenet(pretrained=False)
        model_full.aux1.fc2 = torch.nn.Linear(1024, number_classes)
        model_full.aux2.fc2 = torch.nn.Linear(1024, number_classes)
        model_full.fc = torch.nn.Linear(1024, number_classes)
        model_full.load_state_dict(torch.load(model_weights))
        model_feat = copy.deepcopy(model_full)
        model_feat.fc =torch.nn.Identity()
    elif model_name == 'MobileNetV2':
        model_full = models.mobilenet_v2(pretrained=False)
        model_full.classifier[1] = torch.nn.Linear(1280, number_classes)
        model_full.load_state_dict(torch.load(model_weights))
        model_feat = copy.deepcopy(model_full)
        features = list(model_full.classifier.children())[:-1]
        model_feat.classifier=torch.nn.Sequential(*features)
    elif model_name == 'MobileNetV3L':
        model_full = models.mobilenet_v3_large(pretrained=False)
        model_full.classifier[3] = torch.nn.Linear(1280, number_classes)
        model_full.load_state_dict(torch.load(model_weights))
        model_feat = copy.deepcopy(model_full)
        features = list(model_full.classifier.children())[:-1]
        model_feat.classifier=torch.nn.Sequential(*features)
    elif model_name == 'MobileNetV3S':
        model_full = models.mobilenet_v3_small(pretrained=False) 
        model_full.classifier[3] = torch.nn.Linear(1280, number_classes)
        model_full.load_state_dict(torch.load(model_weights))
        model_feat = copy.deepcopy(model_full)
        features = list(model_full.classifier.children())[:-1]
        model_feat.classifier=torch.nn.Sequential(*features)
    else:
        raise ValueError('Unsupported model. Please choose one of AlexNet, ResNet18, VGG16, \
            SqueezeNet, DenseNet121, DenseNet161, InceptionV3, GoogleNet, MobileNetV2, MobileNetV3L, MobileNetV3S')
        

    model_full.eval() 
    model_feat.eval()
    
    return model_feat, model_full


# General functions

def create_data_series(input_list, index_list, series_name):

    # create a pandas data series
    new_series = pd.Series(input_list, index=index_list, name=series_name)

    return new_series


def dimensionality_reduction(features, n_dimensions=3, drtype='TSNE', perplexity=50, n_iterations=500):

    if drtype == 'TSNE':
        tsne = TSNE(n_components=n_dimensions, verbose=1, perplexity=perplexity, n_iter=n_iterations)
        reduced_coords = tsne.fit_transform(features)
    elif drtype == 'UMAP':
        ufit = umap.UMAP(n_components=n_dimensions)
        reduced_coords = ufit.fit_transform(features)
    else:
        reduced_coords = []
    
    return reduced_coords


def extract_model_metadata(model, image_list, all_class_list, device, transform):
    
    model.to(device)
    pred_list=[]
    conf_list=[]
    for image_filename in image_list:
        image = Image.open(image_filename)
        image = image.convert('RGB')
        img_t = transforms_config.data_transforms[transform](image) 
        img_t = torch.unsqueeze(img_t, 0)
        img_t = img_t.to(device)
        predictions = model(img_t)[0]
        pred_list.append(all_class_list[int(torch.argmax(torch.softmax(predictions.cpu(), dim=0)).detach().numpy())]) 
        conf_list.append(float(torch.max(torch.softmax(predictions.cpu(), dim=0)).detach().numpy()))

    Confidence = create_data_series(conf_list, image_list, 'Confidence')
    Predictions = create_data_series(pred_list, image_list, 'Prediction')

    return (Confidence, Predictions)


def extract_model_features(model, image_list, device, transform, create_series=True):

    model.to(device)
    features_list=[]
    for image_filename in image_list:
        image = Image.open(image_filename)
        image = image.convert('RGB')
        img_t = transforms_config.data_transforms[transform](image)
        img_t = torch.unsqueeze(img_t, 0)
        img_t = img_t.to(device)
        outputs = model(img_t)
        feat_arr = outputs.cpu().detach().numpy()
        features_list.append(feat_arr.flatten())

    if create_series == True:
        features_series = create_data_series(features_list, image_list, 'Features')
    else:
        features_series = features_list

    return features_series   


def extract_embedding_coordinates(model, image_list, device, transform, drtype, stem):

    features_list = extract_model_features(model, image_list, device, transform, create_series=False)

    coords = dimensionality_reduction(features_list, 3, drtype)

    # Only allow for 3 - hard coded for now
    feature_len = len(coords[0,:])
    if feature_len == 3:
        XCoord = create_data_series(coords[:,0], image_list, stem+'0')
        YCoord = create_data_series(coords[:,1], image_list, stem+'1')
        ZCoord = create_data_series(coords[:,2], image_list, stem+'2')
        all_coords = (XCoord, YCoord, ZCoord)
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
    transform_name = config['MODEL VARIABLES']['transform_name']

    # Read in data folders
    # Labelled data
    labelled_folders = config['LABELLED IMAGE FOLDERS']['labelled_folder_list'][1:-1].split(' ')
    labelled_sources = config['LABELLED IMAGE FOLDERS']['labelled_folder_sources'][1:-1].split(' ')
    
    # Unlabelled data
    unlabelled_folders = config['UNLABELLED IMAGE FOLDERS']['unlabelled_folder_list'][1:-1].split(' ')
    unlabelled_sources = config['UNLABELLED IMAGE FOLDERS']['unlabelled_folder_sources'][1:-1].split(' ')

    # Class names
    class_names = config['CLASS NAME FILE']['class_file']

    # Output file
    output_filename = config['OUTPUT FILENAME']['savefile']

    # create image, ground truth label and source list 
    image_list=[]
    class_list=[]
    source_list=[]
    # Loop over labelled folders
    if labelled_folders[0]:
        for ff in range(len(labelled_folders)):
            for image_path in os.listdir(labelled_folders[ff]):
                classfolder = os.path.join(labelled_folders[ff],image_path)
                for image in os.listdir(classfolder):
                    image_list.append(os.path.join(image_path,classfolder,image))
                    class_list.append(image_path)
                    source_list.append(labelled_sources[ff])

    # Read in class names
    all_class_file = open(class_names, 'r')
    all_class_list = all_class_file.read().splitlines()
    all_class_list = [x.strip() for x in all_class_list]
    number_classes = len(set(all_class_list))
    print("Number of classes = ", number_classes)

    if unlabelled_folders[0]:
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
    coords_t = extract_embedding_coordinates(feature_model, image_list, device, transform_name, 'TSNE', 'tsne_coords_') # TSNE Coordinates
    coords_u = extract_embedding_coordinates(feature_model, image_list, device, transform_name, 'UMAP', 'umap_coords_') # UMAP Coordinates
    meta_data = extract_model_metadata(full_model, image_list, all_class_list, device, transform_name) # Prediction, Confidence
    ## TO ADD - if no model, use image intensities for features

    # create pandas dataframe from lists
    all_data = pd.concat([Sources, Labels], axis=1)
    #for ii in len(coords):
    #    all_data = pd.concat([all_data, coords[ii]], axis=1)
    all_data = pd.concat([all_data, coords_t[0], coords_t[1], coords_t[2]], axis=1)
    all_data = pd.concat([all_data, coords_u[0], coords_u[1], coords_u[2]], axis=1)
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
