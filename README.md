# Cursive Text Detection and Multi-Lingual Script Identification in Real-time Natural Scenes

In recent years, researchers have shown a growing interest in identifying written text in natural scene images and videos, driven by the potential for aiding visually impaired individuals or tourists in unfamiliar locations to comprehend information on signs, billboards, and notice boards. The proposed method aims to locate multilingual text in such videos and determine the script, with a focus on English, Hindi, Kannada, and Chinese. An essential project phase involves training the model to recognize cursive text, achieved through a CNN-based YOLOv5 algorithm. The model is trained with a specialized dataset of natural scene images and tested under various conditions, including different backgrounds, fonts, orientations, resolutions, and image distortions. Hyperparameter tuning is conducted to improve cursive text recognition. Experimental results demonstrate the method's effectiveness and robustness. A comparison of model parameters before and after hyperparameter tuning helps identify the optimal configuration for text detection.

![img1](https://github.com/ChandanaGiridhar/Cursive_Text_Detection/blob/main/images/1_cursive_detection_output.png)     ![img2](https://github.com/ChandanaGiridhar/Cursive_Text_Detection/blob/main/images/5_multilingual_detection.png)|

## Introduction ##

The primary objective of this project is to enhance the object detection model's capability to identify and locate cursive text while also categorizing the text into its respective scripts or languages. A key aspect of our focus is to boost the model's effectiveness through hyperparameter tuning, which will significantly improve its ability to recognize cursive text within real-time natural scenes. Various approaches have been proposed by researchers to tackle the challenges of text localization and script identification in natural scene images. Seeri et al. [2] initiated the process using wavelet-based edge features and fuzzy classification. Wang and Shi introduced a novel method incorporating Haar wavelet, edge features, K-means clustering, fuzzy classification, and threshold concepts to pinpoint text and eliminate non-textual regions from images with complex backgrounds. Gupta, Vedaldi, and Zisserman [3] leveraged synthetic data to introduce a fully convolutional regression network (FCRN) for text recognition and bounding box regression, achieving an impressive F-measure of 84.2% on the ICDAR2013 dataset. Kumar et al. [5] put forward an attention-based Convolutional-LSTM network for script identification, extracting both local and global features through the CNN-LSTM framework and weighting them for script identification, achieving an accuracy of 97.75% on four script identification datasets. It is important to note that the success of text localization and script identification is contingent upon both the quality of the training dataset and the machine learning model employed.

In our project, we have accomplished text localization and script identification for Hindi, English, Kannada, and Chinese by training the YOLOv5 model with a robust custom dataset comprising over 2000 images. We have also expanded this dataset to include cursive text detection, and further refined the model's performance through hyperparameter tuning, with a particular focus on enhancing its ability to recognize cursive data. To validate the cursive text detection capability, we manually constructed a dedicated cursive dataset and trained the YOLOv5 model on it.

## Methodology ##

The first phase of the project was creating the custom dataset to train the YOLOv5 Model. We created a custom dataset that includes the following:
- Texts of scripts/languages: English, Chinese, Hindi, Kannada.
- Cursive Style texts.
- Texts found on billboards, Name Boards, and on Streets.
- Captions and texts found in Pre-recorded videos.
- Texts of different orientations.
- Images captured in daylight and at night.
  
The texts found in an image are then labeled using the makesense.ai tool which creates a labeled text file for each image included in the dataset. This custom dataset is given to the model to train and validate.
For text localization and script recognition in a natural scene image/video, the suggested technique includes a deep learning neural network, YOLOv5 based on DarkNet53. It is divided into two stages: training and testing. In a video, the YOLO detects and tracks several targets (objects). The YOLOv5 is taught utilizing a training dataset of natural scene images/videos containing multi-lingual text items throughout the training stage. During the testing step, a natural scene picture or video is fed into the trained YOLO, which produces an image or video with a bounding box for the detected text region and labels the box with the recognized script. During the training process, the YOLO employs binary cross-entropy loss and logistic regression to accomplish category prediction. This enables YOLO to classify a target (object) as having several labels.

![img3](https://github.com/ChandanaGiridhar/Cursive_Text_Detection/blob/main/images/2_overview_of_YOLOv5.png)

## Results & Discussion ##

The trained model is evaluated on multiple graphic text translation pictures and real-time recorded street view images to determine its efficacy. With a frame rate of 45 frames per second, the model can recognize texts in images quickly and identify the script of the localized text. The YOLO recognizes the text by drawing a bounding box around it, classifies the script, and shows the confidence score for each detection.

![img4](https://github.com/ChandanaGiridhar/Cursive_Text_Detection/blob/main/images/3_chinese_detection_output.png)
![img5](https://github.com/ChandanaGiridhar/Cursive_Text_Detection/blob/main/images/6_cursive_detection_output3.png)
## Conclusion ##

We utilized YOLOv5, an efficient object detection algorithm, for text localization and script recognition in natural scene photos using the DarkNet53 Architecture. Despite photo complexity and nuances in Kannada, Hindi, English, and Chinese characters, including cursive text, our model accurately detects text and identifies scripts. We rigorously tested the model in challenging scenarios like varied backgrounds, orientations, fonts, resolutions, and lighting conditions, and further improved its accuracy through hyperparameter tuning. With an expanded dataset and parameter adjustments, our model achieves 95%-99% overall accuracy. However, there's room for improvement in text detection and script recognition, vital for applications like text extraction and translation, benefiting tourists and the visually impaired.

## Links ##

Cursive & Mixed Dataset - [Click Here](https://drive.google.com/drive/folders/1braj6RrGmITpJwGOePEMT8wOXrgxkt6t?usp=drive_link)

Trained YOLOv5 - [Click Here](https://drive.google.com/drive/folders/1EtFsXtgaWZvxMbRKA2sulVyMjIHEIJb3?usp=drive_link)
