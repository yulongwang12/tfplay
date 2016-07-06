import tensorflow as tf

def ROIPoolingLayer(inp, roi, pooled_h, pooled_w):
    '''
        inp: input feature, N x H x W x C
        
        roi: region of interests (roi) bbox
             each one is [batch_idx, start_h, start_w, crop_h, crop_w]
             !!! must be a list
        
        pooled_h/pooled_w: target h/w
        
        out: output feature N x ph x pw x C
    '''
    input_shape = inp.get_shape().as_list()
    inp_h = input_shape[1]
    inp_w = input_shape[2]
    
    roi_num = len(roi)
    
    outlist = []
    for i in xrange(roi_num):
        btch_i, st_h, st_w, cp_h, cp_w =  roi[i]
        
        pool_sz_h = cp_h // pooled_h
        pool_sz_w = cp_w // pooled_w
        
        real_cp_h = pooled_h * pool_sz_h
        real_cp_w = pooled_w * pool_sz_w
        
        cropped_feature = tf.slice(inp, [btch_i, st_h, st_w, 0], [1, real_cp_h, real_cp_w, -1])
        
        outlist.append(tf.nn.max_pool(cropped_feature, ksize=[1, pool_sz_h, pool_sz_w, 1], \
                                      strides=[1, pool_sz_h, pool_sz_w, 1], padding='SAME'))
    
    return tf.concat(0, outlist)