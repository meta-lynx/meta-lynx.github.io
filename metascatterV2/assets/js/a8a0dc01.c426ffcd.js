"use strict";(self.webpackChunkmetascatter=self.webpackChunkmetascatter||[]).push([[597],{3905:function(e,t,a){a.d(t,{Zo:function(){return d},kt:function(){return u}});var n=a(7294);function r(e,t,a){return t in e?Object.defineProperty(e,t,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[t]=a,e}function o(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,n)}return a}function l(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?o(Object(a),!0).forEach((function(t){r(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):o(Object(a)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}function i(e,t){if(null==e)return{};var a,n,r=function(e,t){if(null==e)return{};var a,n,r={},o=Object.keys(e);for(n=0;n<o.length;n++)a=o[n],t.indexOf(a)>=0||(r[a]=e[a]);return r}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(n=0;n<o.length;n++)a=o[n],t.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(r[a]=e[a])}return r}var s=n.createContext({}),p=function(e){var t=n.useContext(s),a=t;return e&&(a="function"==typeof e?e(t):l(l({},t),e)),a},d=function(e){var t=p(e.components);return n.createElement(s.Provider,{value:t},e.children)},c={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},f=n.forwardRef((function(e,t){var a=e.components,r=e.mdxType,o=e.originalType,s=e.parentName,d=i(e,["components","mdxType","originalType","parentName"]),f=p(a),u=r,m=f["".concat(s,".").concat(u)]||f[u]||c[u]||o;return a?n.createElement(m,l(l({ref:t},d),{},{components:a})):n.createElement(m,l({ref:t},d))}));function u(e,t){var a=arguments,r=t&&t.mdxType;if("string"==typeof e||r){var o=a.length,l=new Array(o);l[0]=f;var i={};for(var s in t)hasOwnProperty.call(t,s)&&(i[s]=t[s]);i.originalType=e,i.mdxType="string"==typeof e?e:r,l[1]=i;for(var p=2;p<o;p++)l[p]=a[p];return n.createElement.apply(null,l)}return n.createElement.apply(null,a)}f.displayName="MDXCreateElement"},2121:function(e,t,a){a.r(t),a.d(t,{assets:function(){return d},contentTitle:function(){return s},default:function(){return u},frontMatter:function(){return i},metadata:function(){return p},toc:function(){return c}});var n=a(7462),r=a(3366),o=(a(7294),a(3905)),l=["components"],i={sidebar_position:2},s="Data Preparation: Image Classification",p={unversionedId:"getting-started/image-classification",id:"getting-started/image-classification",title:"Data Preparation: Image Classification",description:"In this section we describe how to create a CSV file from trained image classification tasks, which can be uploaded into Metascatter. We provide scripts for standard classification model architectures (with user-provided weights) for:",source:"@site/docs/getting-started/image-classification.md",sourceDirName:"getting-started",slug:"/getting-started/image-classification",permalink:"/metascatterV2/docs/getting-started/image-classification",tags:[],version:"current",sidebarPosition:2,frontMatter:{sidebar_position:2},sidebar:"tutorialSidebar",previous:{title:"Installation",permalink:"/metascatterV2/docs/getting-started/installation"},next:{title:"Data Preparation: Object Detection",permalink:"/metascatterV2/docs/getting-started/object-detection"}},d={},c=[{value:"Tensorflow",id:"tensorflow",level:2},{value:"Downloads",id:"downloads",level:3},{value:"Quick Start",id:"quick-start",level:3},{value:"Prepare CSV file",id:"prepare-csv-file",level:3},{value:"PyTorch",id:"pytorch",level:2},{value:"Downloads",id:"downloads-1",level:3},{value:"Quick Start",id:"quick-start-1",level:3},{value:"Prepare CSV file",id:"prepare-csv-file-1",level:3}],f={toc:c};function u(e){var t=e.components,i=(0,r.Z)(e,l);return(0,o.kt)("wrapper",(0,n.Z)({},f,i,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"data-preparation-image-classification"},"Data Preparation: Image Classification"),(0,o.kt)("p",null,"In this section we describe how to create a CSV file from trained image classification tasks, which can be uploaded into Metascatter. We provide scripts for standard classification model architectures (with user-provided weights) for:"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"#tensorflow"},"Tensorflow")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"#pytorch"},"PyTorch"))),(0,o.kt)("p",null,"You will ",(0,o.kt)("strong",{parentName:"p"},"only need to edit some configuration files")," to point to your data and model weights."),(0,o.kt)("h2",{id:"tensorflow"},"Tensorflow"),(0,o.kt)("h3",{id:"downloads"},"Downloads"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"Tensorflow classifcation CSV creation script: ",(0,o.kt)("a",{parentName:"li",href:"https://drive.google.com/file/d/11dpkOaAfkaQBZE7ViS_lrxbcwt1KDT5k/view?usp=sharing"},"Download")),(0,o.kt)("li",{parentName:"ul"},"Template configuration file: ",(0,o.kt)("a",{parentName:"li",href:"https://drive.google.com/file/d/1JpwgS0yC58GJHn5gkMQWCa8w9gA-q0q1/view?usp=sharing"},"Download")),(0,o.kt)("li",{parentName:"ul"},"Requirements file: ",(0,o.kt)("a",{parentName:"li",href:"https://drive.google.com/file/d/1ELBWh6ZKXIj6RPsIx2G1cFkdUFOVe_Hi/view?usp=sharing"},"Download"))),(0,o.kt)("p",null,(0,o.kt)("strong",{parentName:"p"},"Usage: ",(0,o.kt)("inlineCode",{parentName:"strong"},"python create_csv_tf.py 'path_to_config_file.ini'"))),(0,o.kt)("h3",{id:"quick-start"},"Quick Start"),(0,o.kt)("p",null,"To create a CSV from a Tensorflow classification model, simply edit the variables in red in the template configuration file:\n",(0,o.kt)("img",{alt:"Tensorflow Quickstart",src:a(9420).Z,width:"633",height:"383"})),(0,o.kt)("h3",{id:"prepare-csv-file"},"Prepare CSV file"),(0,o.kt)("p",null,"We provide scripts to create a CSV file that works with metascatter, given image folders and models. "),(0,o.kt)("p",null,"An example script for Tensorflow classification models can be downloaded here: ",(0,o.kt)("a",{parentName:"p",href:"https://drive.google.com/file/d/11dpkOaAfkaQBZE7ViS_lrxbcwt1KDT5k/view?usp=sharing"},"Tensorflow Classification"),". ",(0,o.kt)("em",{parentName:"p"},"You should not need to edit this file"),". "),(0,o.kt)("p",null,"This requires the following Python3 libraries:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"tensorflow==2.8.0\npillow>=9.1.0\npandas>=1.4.2\nsklearn\n")),(0,o.kt)("p",null,"The following models can be used, using either ImageNet or your own pre-trained weights:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"MobileNet, MobiletNetV2\nResNet50, ResNet101V2, ResNet152V2, ResNet50V2\nVGG16, VGG19\nDenseNet121, DenseNet169, DenseNet201\nEfficientNetB0, EfficientNetB7\nEfficientNetV2L, EfficientNetV2M, EfficientNetV2S\nInceptionV2, InceptionV3\n")),(0,o.kt)("p",null,"You will need to supply a configuration file: ",(0,o.kt)("a",{parentName:"p",href:"https://drive.google.com/file/d/1JpwgS0yC58GJHn5gkMQWCa8w9gA-q0q1/view?usp=sharing"},"Download Template Config File")),(0,o.kt)("p",null,"Usage: ",(0,o.kt)("strong",{parentName:"p"},(0,o.kt)("inlineCode",{parentName:"strong"},"python create_csv_tf.py 'path_to_config_file.ini'"))),(0,o.kt)("p",null,"The ",(0,o.kt)("inlineCode",{parentName:"p"},"create_csv_tf.py")," script should not need to be changed. Edit the configuration file as below:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"[MODEL VARIABLES]\nmodel_name = MobileNet\nmodel_weights = /Path/to/image/weights.h5\nimage_size = 224\n")),(0,o.kt)("p",null,"Please provide one of the models listed above and a path to the trained model weights. Also include the height/width of the images needed by the model (default 224)."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"[LABELLED IMAGE FOLDERS] \nlabelled_folder_list: [/Path/to/folder1 /Path/to/folder/2 /Path/to/folder/3]\nlabelled_folder_sources: [Labelled_source_1 Labelled_source_2 Labelled_source 3]\n# Images should be arranged in folders according to class:\n#    Folder->Class->Image. \n# For multiple locations, please separate folders and sources by a space.\n")),(0,o.kt)("p",null,"Inlcude a list of folders which store the labelled images you want to use. The folder structure of each image should be in Tensorflow classification format:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"\u251c\u2500\u2500 Image folder:\n|   \u251c\u2500\u2500 Class 1 Folder:\n|   |   \u251c\u2500\u2500 Image1.png\n|   |   \u251c\u2500\u2500 Image2.png\n|   |   \u2514\u2500\u2500 Image3.png\n|   \u251c\u2500\u2500 Class 2 Folder:\n|   |   \u251c\u2500\u2500 Image4.png\n|   |   \u2514\u2500\u2500 Image5.png\n")),(0,o.kt)("p",null,"You can provide several folders, e.g. if you have different folders for ",(0,o.kt)("inlineCode",{parentName:"p"},"TRAINING"),", ",(0,o.kt)("inlineCode",{parentName:"p"},"TESTING")," and ",(0,o.kt)("inlineCode",{parentName:"p"},"VALIDATION")," images. You can reference these by entereding a corresponding name in the field ",(0,o.kt)("inlineCode",{parentName:"p"},"labelled_folder_sources"),". Please ensure there are the same number of source names as folders provided. The folders and names should be separated by a ",(0,o.kt)("strong",{parentName:"p"},"space"),". "),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"[UNLABELLED IMAGE FOLDERS]\nunlabelled_folder_list: [/Path/to/folder1 /Path/to/folder2]\nunlabelled_folder_sources: [Unlabelled_source_1 Unlabelled_source_2]\n# Unordered image folder structure: Folder->Image. \n# For multiple locations, please separate folders and sources by a space.\n")),(0,o.kt)("p",null,"Similarly, you can include one or many folders for unlabelled data. Leave blank if there are no such folders. The structure of these folders should be:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"\u251c\u2500\u2500 Image Folder:\n|   \u251c\u2500\u2500 Image1.png\n|   \u251c\u2500\u2500 Image2.png\n|   \u2514\u2500\u2500 Image3.png\n")),(0,o.kt)("p",null,"Finally, enter the filename and path of the output ",(0,o.kt)("inlineCode",{parentName:"p"},"csv")," file. "),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"[OUTPUT FILENAME]\nsavefile = /Path/to/outputfile.csv\n")),(0,o.kt)("p",null,"This works with standard architectures of the models named above with either ImageNet or retrained weights. For bespoke architectures, please see ",(0,o.kt)("a",{parentName:"p",href:"#data-preparation"},"Data Preparation"),"."),(0,o.kt)("h2",{id:"pytorch"},"PyTorch"),(0,o.kt)("h3",{id:"downloads-1"},"Downloads"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"PyTorch classifcation CSV creation script: ",(0,o.kt)("a",{parentName:"li",href:"https://drive.google.com/file/d/1QvV_30O-8Plpt6J_w3E-LaaJDg25X-C8/view?usp=sharing"},"Download")),(0,o.kt)("li",{parentName:"ul"},"Template configuration file: ",(0,o.kt)("a",{parentName:"li",href:"https://drive.google.com/file/d/1JpwgS0yC58GJHn5gkMQWCa8w9gA-q0q1/view?usp=sharing"},"Download")),(0,o.kt)("li",{parentName:"ul"},"Template transforms file: ",(0,o.kt)("a",{parentName:"li",href:"https://drive.google.com/file/d/1FxvsQLbtKWYN5w1yCYcxNUC0LX0lcMzt/view?usp=sharing"},"Download")),(0,o.kt)("li",{parentName:"ul"},"Requirements file: ",(0,o.kt)("a",{parentName:"li",href:"https://drive.google.com/file/d/1XRL_3RrTWnQLJmtRMENPVLiyWb7km4Et/view?usp=sharing"},"Download"))),(0,o.kt)("p",null,(0,o.kt)("strong",{parentName:"p"},"Usage: ",(0,o.kt)("inlineCode",{parentName:"strong"},"python create_csv_torch.py 'path_to_config_file.ini'"))),(0,o.kt)("h3",{id:"quick-start-1"},"Quick Start"),(0,o.kt)("p",null,"To create a CSV from a Pytorch classification model, simply edit the variables in red in the template configuration file:\n",(0,o.kt)("img",{alt:"PyTorch Quickstart",src:a(6019).Z,width:"638",height:"448"})),(0,o.kt)("h3",{id:"prepare-csv-file-1"},"Prepare CSV file"),(0,o.kt)("p",null,"We provide scripts to create a CSV file that works with metascatter, given image folders and models. "),(0,o.kt)("p",null,"An example script for PyTorch classifcation models can be downloaded here: ",(0,o.kt)("a",{parentName:"p",href:"https://drive.google.com/file/d/1QvV_30O-8Plpt6J_w3E-LaaJDg25X-C8/view?usp=sharing"},"PyTorch script download"),". ",(0,o.kt)("em",{parentName:"p"},"You should not ordinarily need to edit this file.")," "),(0,o.kt)("p",null,"The following Python3 libraries are required:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"torch\ntorchvision\npillow>=9.1.0\npandas>=1.4.2\nsklearn\numap-learn\n")),(0,o.kt)("p",null,"The following models can be used with your own trained weights:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"AlexNet, ResNet18, VGG16, SqueezeNet, DenseNet161, InceptionV3, GoogleNet, MobileNetV2, MobileNetV3L, MobileNetV3S\n")),(0,o.kt)("p",null,"You will need to supply a configuration file and a file describing the transforms for inference, for which templates can be found below:"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://drive.google.com/file/d/1JpwgS0yC58GJHn5gkMQWCa8w9gA-q0q1/view?usp=sharing"},"Template configuration file")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://drive.google.com/file/d/1FxvsQLbtKWYN5w1yCYcxNUC0LX0lcMzt/view?usp=sharing"},"Template transforms file"))),(0,o.kt)("p",null,"Usage: ",(0,o.kt)("strong",{parentName:"p"},(0,o.kt)("inlineCode",{parentName:"strong"},"python create_csv_torch.py 'path_to_config_file.ini'"))),(0,o.kt)("p",null,"The ",(0,o.kt)("inlineCode",{parentName:"p"},"create_csv_torch.py")," script should not be changed. Edit the configuration file as below."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"[MODEL VARIABLES]\nmodel_name = AlexNet\nmodel_weights = /path/to/model/weight/file.pth\ntransform_name = inference\n# Should correspond to transforms_config.py\n")),(0,o.kt)("p",null,"Please use one of the standard model architectures listed above and provide the path to your trained weights. The ",(0,o.kt)("inlineCode",{parentName:"p"},"transform_name")," field should correspond to the name given in ",(0,o.kt)("inlineCode",{parentName:"p"},"transforms_config.py"),". "),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"[LABELLED IMAGE FOLDERS] \nlabelled_folder_list: [/path/to/folder/of/labelled/images/1/ /path/to/folder/of/labelled/images/2/ /path/to/folder/of/labelled/images/3/]\nlabelled_folder_sources: [Name_of_source_of_folder_1 Name_of_source_of_folder_2 Name_of_source_of_folder_3] \n# Images should be arranged in folders according to class: \n# Folder->Class->Image. For multiple locations, please separate \n# folders and sources by a space.\n")),(0,o.kt)("p",null,"Inlcude a list of folders which store the labelled images you want to use. The folder structure of each image should be in the following class format:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"\u251c\u2500\u2500 Image folder:\n|   \u251c\u2500\u2500 Class 1 Folder:\n|   |   \u251c\u2500\u2500 Image1.png\n|   |   \u251c\u2500\u2500 Image2.png\n|   |   \u2514\u2500\u2500 Image3.png\n|   \u251c\u2500\u2500 Class 2 Folder:\n|   |   \u251c\u2500\u2500 Image4.png\n|   |   \u2514\u2500\u2500 Image5.png\n")),(0,o.kt)("p",null,"You can provide several folders, e.g. if you have different folders for ",(0,o.kt)("inlineCode",{parentName:"p"},"TRAINING"),", ",(0,o.kt)("inlineCode",{parentName:"p"},"TESTING")," and ",(0,o.kt)("inlineCode",{parentName:"p"},"VALIDATION")," images. You can reference these by entering a corresponding name in the field ",(0,o.kt)("inlineCode",{parentName:"p"},"labelled_folder_sources"),". Please ensure there are the same number of source names as folders provided. The folders and names should be separated by a ",(0,o.kt)("strong",{parentName:"p"},"space"),". "),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"[UNLABELLED IMAGE FOLDERS]\nunlabelled_folder_list: [/Path/to/folder1 /Path/to/folder2]\nunlabelled_folder_sources: [Unlabelled_source_1 Unlabelled_source_2]\n# Unordered image folder structure: Folder->Image. \n# For multiple locations, please separate folders and sources by a space.\n")),(0,o.kt)("p",null,"Similarly, you can include one or many folders for unlabelled data. Leave blank if there are no such folders. As there are no classes, the structure of these folders should be:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"\u251c\u2500\u2500 Image Folder:\n|   \u251c\u2500\u2500 Image1.png\n|   \u251c\u2500\u2500 Image2.png\n|   \u2514\u2500\u2500 Image3.png\n")),(0,o.kt)("p",null,"In order to output class names (instead of numbers) to the CSV, you will need to provide a class list file."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"[CLASS NAME FILE]\nclass_file = /path/to/file/with/class/names.txt\n")),(0,o.kt)("p",null,"The list of classes should be in order corresponding to the output of the model:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"Class0\nClass1\nClass2\n...\nClassN\n")),(0,o.kt)("p",null,"Finally, enter the filename and path of the output ",(0,o.kt)("inlineCode",{parentName:"p"},"csv")," file. "),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"[OUTPUT FILENAME]\nsavefile = /Path/to/outputfile.csv\n")),(0,o.kt)("p",null,"This works with standard architectures of the models named above with either ImageNet or retrained weights. For bespoke architectures, please see ",(0,o.kt)("a",{parentName:"p",href:"#data-preparation"},"Data Preparation"),"."))}u.isMDXComponent=!0},6019:function(e,t,a){t.Z=a.p+"assets/images/pytorch_quickstart-7c26cb1137a4e68bac09b25fa77e5f0d.png"},9420:function(e,t,a){t.Z=a.p+"assets/images/tensorflow_quickstart-d15ac0edf2709348c10cafb2b2ff10f4.png"}}]);