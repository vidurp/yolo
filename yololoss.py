import tensorflow as tf

# Add a tf.function decorator, allows for graph execution
@tf.function
def CalcIOU( BBoxTrue, BBoxPred ):
    '''
    Calculates Intersection over union IOU
    Supplied bounding boxes have general shape [ N , 4 ] where each element
    is of format x,y,w,h

    Args:
        BBoxTrue - Truth Bounding Box(es)
        BBoxPred - Predicted Bounding Box(es)

    Returns:
        A tensor of shape (N, ) containing IOU values
    '''
    AreaBBoxTrue = BBoxTrue[...,2] * BBoxTrue[...,3]
    AreaBBoxPred = BBoxPred[...,2] * BBoxPred[...,3]

    # top left coordinates of intersection
    xtop = tf.maximum( BBoxTrue[...,0], BBoxPred[...,0] )
    ytop = tf.maximum( BBoxTrue[...,1], BBoxPred[...,1] )

    # bottom right coordinates of intersection
    xbottom =  tf.minimum( BBoxTrue[...,0] + BBoxTrue[...,2], BBoxPred[...,0] + BBoxPred[...,2] )
    ybottom =  tf.minimum( BBoxTrue[...,1] + BBoxTrue[...,3], BBoxPred[...,1] + BBoxPred[...,3] )

    AreaInterSection = tf.maximum( xbottom - xtop, 0.0 ) * tf.maximum( ybottom - ytop, 0.0 )

    AUnion = AreaBBoxPred + AreaBBoxTrue - AreaInterSection +  + 0.0001
    iou = AreaInterSection / AUnion
    return iou
    
@tf.function
def YoloLoss( Ytrue , Ypred ):
    """
    Loss Function for YOLO Algorithm
    Args:
        Ytrue: Truth Labels ( Batch x 7 x 7 x 17 )
        YPred: Predicted Labels ( Batch x 7 x 7 x 17 ) 
        
        Two predictors pers cell. Each pred is a BBOX + Confidence + 7 classes
        17 -> ( 4 + 1 ) x 2 + 7
         
    Returns: Loss Value
    """
    ClassTrue = tf.reshape( Ytrue[...,10:], shape=(-1,7) )
    ClassPred = tf.reshape( Ypred[...,10:], shape=(-1,7) )

    # change 7x7xN into 49 x N 1D samples
    BBox1True = tf.reshape( Ytrue[...,0:4], shape=(-1,4) )
    Conf1True = tf.reshape( Ytrue[...,4], shape=(-1,) )
    BBox2True = tf.reshape( Ytrue[...,5:9], shape=(-1,4 ) )
    Conf2True = tf.reshape( Ytrue[...,9], shape=(-1,) )

    BBox1Pred = tf.reshape( Ypred[...,0:4], shape=(-1,4) )
    Conf1Pred = tf.reshape( Ypred[...,4], shape=(-1,) )
    BBox2Pred = tf.reshape( Ypred[...,5:9], shape=(-1,4) )
    Conf2Pred = tf.reshape( Ypred[...,9], shape=(-1,) )


    # BBox variables now have shape ( Batch x 49 , 4 ) for iou calculations
    iou1 = Conf1True * CalcIOU( BBox1True, BBox1Pred )
    iou2 = Conf2True * CalcIOU( BBox2True, BBox2Pred )

    # Find Responsible Predictor
    ValidPredictor1 = tf.cast( tf.math.greater( iou1, iou2 ), dtype=tf.float32 )
    ValidPredictor2 = tf.cast( tf.math.greater( iou2, iou1 ), dtype=tf.float32 )

    # confidence loss from both predictors
    ConfidenceLossPredictor1 = ValidPredictor1 * tf.square( Conf1True - Conf1Pred )
    ConfidenceLossPredictor2 = ValidPredictor2 * tf.square( Conf2True - Conf2Pred )
    Loss = tf.reduce_mean( ConfidenceLossPredictor1, axis = 0 )
    Loss += tf.reduce_mean( ConfidenceLossPredictor2, axis = 0 )

    # Classification Loss
    ClassificationLoss = Conf1True * tf.cast(tf.square( tf.argmax(ClassTrue,axis=1) -  tf.argmax(ClassPred,axis=1) ), dtype=tf.float32 )

    Loss += tf.reduce_sum( ClassificationLoss, axis = 0 )

    # Bounding Box Loss
    BBoxLossPredictor1 = ValidPredictor1 * tf.reduce_sum( tf.square( BBox1True[:,:2] - BBox1Pred[:,:2] ) +
                                                         tf.square( tf.sqrt(BBox1True[:,2:]) - tf.sqrt(BBox1Pred[:,2:]) ), axis = 1 )

    BBoxLossPredictor2 = ValidPredictor2 * tf.reduce_sum( tf.square( BBox2True[:,:2] - BBox2Pred[:,:2] ) +
                                                          tf.square( tf.sqrt(BBox2True[:,2:]) - tf.sqrt(BBox2Pred[:,2:]) ), axis = 1 )


    Loss += tf.reduce_sum( BBoxLossPredictor1, axis = 0 )
    Loss += tf.reduce_sum( BBoxLossPredictor2, axis = 0 )

    return Loss
