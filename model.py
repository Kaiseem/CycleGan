import tensorflow as tf
from ops import lrelu,conv, instance_norm

def discriminator(image, hidden_num=64, reuse=False, name="discriminator"):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        h = conv(image, 4, hidden_num, 2, 'd_h0_conv', act_func=lrelu)
        h1 = lrelu(instance_norm(conv(h, 4, hidden_num * 2, 2, 'd_h1_conv'), 'd_bn1'))
        h2 = lrelu(instance_norm(conv(h1, 4, hidden_num * 4, 2, 'd_h2_conv'), 'd_bn2'))
        h3 = lrelu(instance_norm(conv(h2, 4, hidden_num * 8, 1, 'd_h3_conv'), 'd_bn3'))
        h4 = conv(h3, 4, 1, 1, 'd_h4_conv')
    return h4


def generator(image, hidden_num=64, reuse=False, name="generator"):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, name='res'):
            ks=3
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv(y, ks, hidden_num * 4, 1, name + '_c1', padding='VALID'), name + '_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv(y, ks, hidden_num * 4, 1, name + '_c2', padding='VALID'), name + '_bn2')
            return y + x
        def resize_convolution(x,size,hnum,name='rszconv'):
            y=tf.image.resize_images(x,(size,size),method=1)
            y=tf.nn.relu(instance_norm(conv(y,3,hnum,1,name+'_rszconv'),name+'_bn'))
            return y
        def deepwise_seperable_convolutional_residule_block(x,name):#deepwise_seperable_convolutional_residule_block
            #tf.nn.depthwise_conv2d(input,filter,strides,padding,rate=None,name=None,data_format=None)
            #filter [filter_height, filter_width, in_channels, channel_multiplier]
            #output [batch, out_height, out_width, in_channels * channel_multiplier]
            #tf.layers.separable_conv2d(input,filter=outputchannel,kernel=(3,3)=tf.nn.depthwise_conv2d+conv1*1
            ks=3
            p = int((ks - 1) / 2)
            in_channel=hidden_num * 4
            with tf.variable_scope(name+'_deconv1'):
                y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
                y= instance_norm(tf.layers.separable_conv2d(y, in_channel,(ks,ks), padding='VALID',use_bias=False), name + '_bn1')
            with tf.variable_scope(name+'_deconv2'):
                y = tf.pad(y, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
                y= instance_norm(tf.layers.separable_conv2d(y, in_channel,(ks,ks), padding='VALID',use_bias=False), name + '_bn2')
            with tf.variable_scope(name+'_deconv3'):
                y = tf.pad(y, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
                y= instance_norm(tf.layers.separable_conv2d(y, in_channel,(ks,ks), padding='VALID',use_bias=False), name + '_bn3')
            return x+y

        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(conv(c0, 7, hidden_num, 1, 'g_e1_c', padding='VALID'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv(c1, 3, hidden_num * 2, 2, 'g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv(c2, 3, hidden_num * 4, 2, 'g_e3_c'), 'g_e3_bn'))
        func=deepwise_seperable_convolutional_residule_block
        r1 = func(c3, 'g_r1')
        r2 = func(r1, 'g_r2')
        r3 = func(r2, 'g_r3')
        r4 = func(r3, 'g_r4')
        r5 = func(r4, 'g_r5')
        r6 = func(r5, 'g_r6')
        r7 = func(r6, 'g_r7')
        r8 = func(r7, 'g_r8')
        r9 = func(r8, 'g_r9')

        #to avoid checkerboard, using kernel size/stride=constant deconv (sub-pixel convolution) or using
        d1=resize_convolution(r9,128,hidden_num*2,name='g_rs1')
        d2=resize_convolution(d1,256,hidden_num,name='g_rs2')
        #ASPP
        #d1 = tf.nn.relu(instance_norm(tf.layers.conv2d_transpose(r9, filters=hidden_num * 2, kernel_size=3, strides=2, padding='same', activation=None,name= 'g_d1_dc'), 'g_d1_bn'))
        #d2 = tf.nn.relu(instance_norm(tf.layers.conv2d_transpose(d1, filters=hidden_num , kernel_size=3, strides=2, padding='same', activation=None,name= 'g_d2_dc'), 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        out = tf.nn.tanh(conv(d2, 7, 3, 1, 'g_out', padding='VALID'))#256*256
    return out

def MSE_loss(in_, target):#mse mean-square error minimum squared error MMSE=tf.sqrt(tf.reduce_mean((in_-target)**2))
    return tf.reduce_mean((in_-target)**2)

def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))