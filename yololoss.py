import tensorflow as tf

# Add a tf.function decorator, allows for graph execution
@tf.function
def CalcIOU( BBoxTrue, BBoxPred ):
    """
    Function that Calculates Intersection Over Union between two 
    Bounding Boxes. Bounding Boxes arguments are supplied as tuples
    ( x, y, w, h )
    
    Args:
        BBoxTrue: Truth Value Bounding Box
        BBoxPred: Predicted Bounding Box
        
    Returns: IOU value, if there is no overlap, 
             returns 0.01 as a base floor
    """
    xtrue = BBoxTrue[0]
    ytrue = BBoxTrue[1]
    wtrue = BBoxTrue[2]
    htrue = BBoxTrue[3]
    xpred = BBoxPred[0]
    ypred = BBoxPred[1]
    wpred = BBoxPred[2]
    hpred = BBoxPred[3]

    AUnion = 0
    AInter = 0

    if ( wpred == 0 or hpred == 0 or wtrue == 0 or htrue == 0 ):
        iou = tf.constant(0.01, dtype=tf.float32)
    # check for no overlap
    elif ( ( (( xpred + wpred ) <  xtrue ) or ( (xpred + wpred) > (xtrue + wtrue) ) ) and
         ( (( ypred + hpred ) <  ytrue ) or ( (ypred + hpred) > (ytrue + htrue) ) ) ):
        iou = tf.constant(0.01, dtype=tf.float32)
    else:
        # Intersection
        Atrue = tf.math.multiply( wtrue , htrue )
        Apred = tf.math.multiply( wpred , hpred )
        AInter = 0

        xinter = tf.math.maximum( xtrue, xpred )
        yinter = tf.math.maximum( ytrue, ypred )
        xinter2 = tf.math.minimum( xtrue + wtrue, xpred + wpred )
        yinter2 = tf.math.minimum( ytrue + htrue, ypred + hpred )
        AInter = tf.math.abs( tf.math.multiply( ( xinter - xinter2 ) , ( yinter - yinter2 ) ) )

        AUnion = tf.math.add(Atrue , Apred )
        AUnion = tf.math.subtract( AUnion,  AInter )
        iou = tf.math.divide( AInter , AUnion )

    return (tf.floor(iou * 100) / 100)
    
    
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