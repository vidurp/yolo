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

    AUnion = AreaBBoxPred + AreaBBoxTrue - AreaInterSection
    iou = AreaInterSection / AUnion
    return iou
    
@tf.function
def YOLOLoss( Ytrue , Ypred ):
    """
    Loss Function for YOLO Algorithm
    Args:
        Ytrue: Truth Labels
        YPred: Predicted Labels
        
    Returns: Loss Value
    """
    PredT = tf.reshape( tf.convert_to_tensor( Ypred ), ( 7,7,17 ) )
    TrueT = tf.reshape( tf.convert_to_tensor( Ytrue ), ( 7,7,17 ) )

    CategoricalCrossEntropyFunc = tf.keras.losses.CategoricalCrossentropy( )

    Loss = tf.constant(0, dtype=tf.float32)

    for i,j in itertools.product(range(7),range(7)):
        GridCellIOU = tf.constant(0, dtype=tf.float32)
        BBoxLoss = tf.constant(0, dtype=tf.float32)
        ConfLoss = tf.constant(0, dtype=tf.float32)
        ClassificationLoss = tf.constant(0, dtype=tf.float32)

        for b in range(2):
            x_true = TrueT[i,j,(b*5)]
            y_true = TrueT[i,j,(b*5)+1]
            w_true = tf.abs(TrueT[i,j,(b*5)+2])
            h_true = tf.abs(TrueT[i,j,(b*5)+3])

            Present =  TrueT[i,j,(b*5)+4]
            c_true = TrueT[i,j,10:]


            x_pred = PredT[i,j,(b*5)]
            y_pred = PredT[i,j,(b*5)+1]
            w_pred = tf.abs(PredT[i,j,(b*5)+2])
            h_pred = tf.abs(PredT[i,j,(b*5)+3])

            Conf =  PredT[i,j,(b*5)+4]
            c_pred = PredT[i,j,10:]

            TrueBBox = ( x_true, y_true, w_true, h_true )
            PredBBox = ( x_pred, y_pred, w_pred, h_pred )


            # Bounding Box Loss is specified by IOU metric
            LocalIOU = CalcIOU( TrueBBox, PredBBox )

            #if we have better
            if tf.greater( LocalIOU, GridCellIOU ):
                GridCellIOU = LocalIOU
                BBoxLoss =  tf.square( x_true - x_pred ) + tf.square( y_true - y_pred ) +  tf.square( w_true - w_pred ) + tf.square( h_true - h_pred )
                BBoxLoss = tf.multiply( BBoxLoss, Present )
                # Confidence Loss
                ConfLoss = tf.square( Present - Conf )
                ConfLoss = tf.multiply( ConfLoss, Present )

        # Classification Loss is only applied if there is an object in this grid cell
        if tf.greater( ConfLoss, 0.0 ):
            ClassificationLoss = CategoricalCrossEntropyFunc( c_true, c_pred )



        Loss = tf.add( Loss, BBoxLoss )
        Loss = tf.add( Loss, ConfLoss )
        Loss = tf.add( Loss, ClassificationLoss )
    return Loss