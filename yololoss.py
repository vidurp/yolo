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