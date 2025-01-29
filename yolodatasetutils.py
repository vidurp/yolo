import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
import itertools
import tensorflow as tf

def YOLOViewPredImage( img, labels, ClassLabels, imgshape = ( 224, 224, 3 ), S = 7, B = 2, ConfidenceThreshold = 0.5 ):
    """
    Function draws and annotates image pixels and labels for a YOLO dataset created by
    CreateYOLODataSet()

    Args:
        img         - image tensor
        labels      - label tensor
        ClassLabels - string List of image classes
        imgshape    - YOLO Image shape. Default ( 224, 224, 3 )
        S           - Num Grid Cells Per Dimension. Default 7
        B           - Num Predictions Per Grid Cell. Default 2
        ConfidenceThreshold - Draw Bounding box if Threshold is greater
                             than this value. Default 0.5

    Returns:
        None
    """
    C = len(ClassLabels)

    if img.ndim == 4 and img.shape[0] == 1:
        img = tf.squeeze(img)

    # extract image
    img = (img.numpy()*255).astype(np.uint8)

    # Decode and overlay labels
    labels = tf.reshape( labels, shape=( S, S ,((B*5) + C) ) )

    for i,j,b in itertools.product(range(7),range(7),range(2)):
        Conf = labels[i,j,(b*5) + 4]
        if Conf.numpy() > ConfidenceThreshold:
            xmin = (labels[i,j,(b*5)]*imgshape[0]).numpy().astype(np.uint32)
            ymin = (labels[i,j,(b*5)+1]*imgshape[1]).numpy().astype(np.uint32)
            width = (labels[i,j,(b*5)+2]*imgshape[0]).numpy().astype(np.uint32)
            height = (labels[i,j,(b*5)+3]*imgshape[1]).numpy().astype(np.uint32)
            topleft = ( xmin, ymin )
            bottomright = ( xmin+width, ymin+height )
            cv2.rectangle(img, topleft, bottomright,  (255, 0, 0), 2)

            # Annotation Text
            LabelText = ClassLabels[tf.argmax(labels[i,j,(B*5):],0)] + ' ' + str(Conf.numpy())
            cv2.putText( img,
                         LabelText,
                         ( xmin, ymin-4 ),
                         cv2.FONT_HERSHEY_SIMPLEX,
                         0.3,
                         (255,0,0),
                         1)

    plt.imshow(img)
    plt.axis(False)

def CreateOutputTensor( jsondata, idx, ImgScale, ClassLabels, C, S, B ): 
    """
    Creates a YOLO friendly numpy PREDICTION/TRUTH tensor from input files
    Notes:
    The shape of the output tensor
        1. Image is divided into S x S cells
        2. Each cell makes upto B predictions 
        3. Each prediction is a 5-tuple ( xmin, ymin, width, height, Confidence )
        4. YOLO also makes a classification prediction one of C classes
    So Tensor shape = S x S x ( 5B + C ) 

    Args:
        jsondata: json object of input dataset
 
        idx:      index of object in dataset

        ImgScale: tuple representing pixel size of image 

        ClassLabels: list containing string class labels

        C:        Number of Prediction Classes

        S:        Number of Grid cells along each image dimension

        B:        Number of predictions per grid cell

    Returns:
        TruthTensor: numpy array of shape ( S, S, ( 5B + C )) 
    """
    TruthTensor = np.zeros(shape=(S,S,((5*B) + C)))

    NumDetections = jsondata['files'][idx]['numobjects']

    #Grid Cell size in Pixels
    xGridSize = ImgScale[0] // S
    yGridSize = ImgScale[1] // S

    # true image size
    imgwidth = int(jsondata['files'][idx]['imagesize']['width'])
    imgheight  = int(jsondata['files'][idx]['imagesize']['height'])

    # Calculate x,y coordinate scale percentage
    xscale = ImgScale[0] / imgwidth 
    yscale = ImgScale[1] / imgheight

    # Mechanism to Track Num Predictions Per Cell
    NumPredictionsPerGridCell = np.zeros(shape=(S,S),dtype=np.uint32)

    for obj in range(int(NumDetections)):
        # Object Class 
        ImageClass = jsondata['files'][idx]['object'][obj]['label']
        ClassIndex = ClassLabels.index(ImageClass)

        # Bounding Box
        xmin = int(jsondata['files'][idx]['object'][obj]['bndbox']['xmin'])
        ymin = int(jsondata['files'][idx]['object'][obj]['bndbox']['ymin'])
        xmax = int(jsondata['files'][idx]['object'][obj]['bndbox']['xmax'])
        ymax = int(jsondata['files'][idx]['object'][obj]['bndbox']['ymax'])

        # Rescale bounding box from True Image Dimensions to ImgScale
        width = ( xmax - xmin ) * xscale
        height = ( ymax - ymin ) * yscale
        xmin = xmin * xscale
        ymin = ymin * yscale 
        xmax = xmax * xscale
        ymax = ymax * yscale 

        # Find the grid cell this is assigned to
        xcenter = int((( xmin + xmax ) / 2 ) // xGridSize)
        ycenter = int((( ymin + ymax ) / 2 ) // yGridSize)

        # Normalize The bounding box to range [0,1]
        xmin /= ImgScale[0]
        ymin /= ImgScale[1]
        width /= ImgScale[0]
        height /= ImgScale[1]
            
        # Each Grid Cell can make at most B predictions
        NumPredForCurrentGridCell = NumPredictionsPerGridCell[xcenter,ycenter]
        if NumPredForCurrentGridCell <= B:
            # Insert Predicted Class
            ClassVector = np.zeros(C)
            ClassVector[ClassIndex] = 1.0
            TruthTensor[xcenter,xcenter,(5*B):] = ClassVector

            # Insert BBox into corret Prediction Slot
            TruthTensor[xcenter,xcenter,(5*NumPredForCurrentGridCell): ((5*NumPredForCurrentGridCell)+ 5) ] = [ xmin, ymin, width, height, 1.0 ]        

            # Book-keeping, keep track of num detections per cell
            NumPredictionsPerGridCell[xcenter,ycenter] += 1 

    # return Tensor
    return TruthTensor
    
def CreateYOLODataSet( JSONFilePath, ImgScale = (224,224), S = 7, B = 2 ):
    """
    Creates a Image DataSet suitable for YOLO Object detection
    Redmon et.al https://arxiv.org/abs/1506.02640

    Args:
        JSONFilePath - JSON formated Image Descriptor
                       The File is created by pascalvoc.py
        See:
        https://raw.githubusercontent.com/vidurp/generic/refs/heads/main/pascalvoc.py
        
        Example JSON File
        extras\voc2005.json
        
        ImgScale - Images are scaled to default 224,224

        S - Images are divided into SxS uniform cells

        B - Max Number of BBoxes prediced by each cell

    Returns:
        ImgDataTensor - numpy list of resized images

        TruthTensor   - numpy list of truth tensors    
    """
    FileHandle = open(JSONFilePath, 'r')
    jsondata = json.load(FileHandle)
    FileHandle.close()
    ImgDataTensor = []
    TruthTensor = []

    # Num Classes
    C = len(jsondata['labels'])
    # Extract Class Labels as a list
    ClassLabels = [ indx['label'] for indx in jsondata['labels']]

    for idx in range(int(jsondata['numimages'])):
        ImgData = plt.imread(jsondata['files'][idx]['filepath'])
        # convert monochrome image to color, i.e. 3 channels
        if ImgData.ndim != 3:
            ImgData = cv2.cvtColor(ImgData, cv2.COLOR_GRAY2BGR)
        # PNG images may have a ALPHA/transparency channel shape = ( x, y, 4),
        # suppress it
        if ImgData.ndim == 3 and ImgData.shape[2] != 3:
            ImgData = ImgData[:,:,:3]
        ImgData = cv2.resize( ImgData, dsize=ImgScale, interpolation=cv2.INTER_AREA )
     
        ImgDataTensor.append( ImgData )

        # Output Tensor ( YOLO formatted )
        YoloTensor = CreateOutputTensor( jsondata, idx, ImgScale, ClassLabels, C, S, B )
        TruthTensor.append( YoloTensor )

    return ImgDataTensor, TruthTensor