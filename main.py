import tensorflow as tf
from train import cyclegan
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('epoch', 200, "total epoch")
tf.app.flags.DEFINE_integer('epoch_step', 100, 'lr decay')
tf.app.flags.DEFINE_integer('batch_size', 1, 'batch size')
tf.app.flags.DEFINE_integer('train_size',1e8,'')
tf.app.flags.DEFINE_integer('load_size',256,'')
tf.app.flags.DEFINE_integer('image_size',512,'')
tf.app.flags.DEFINE_integer('fine_size',256,'')
tf.app.flags.DEFINE_integer('max_size',50,'')
tf.app.flags.DEFINE_float('lr',0.0002,'')
tf.app.flags.DEFINE_float('beta1',0.5,'')
tf.app.flags.DEFINE_float('L1_lambda',10.0,'')
tf.app.flags.DEFINE_boolean('is_continue_train',True,'')
tf.app.flags.DEFINE_string('save_name', 'D:/GAN/BEGAN/saved/model.ckpt', "saved parameters")
tf.app.flags.DEFINE_string('Apath', 'G:/cyclegan/vangogh2photo/trainA/', "")
tf.app.flags.DEFINE_string('Bpath', 'G:/cyclegan/vangogh2photo/trainB/', "")
tf.app.flags.DEFINE_string('Atpath', 'G:/cyclegan/vangogh2photo/testA/', "")
tf.app.flags.DEFINE_string('Btpath', 'G:/cyclegan/vangogh2photo/testB/', "")
def main(_):
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth=True
    with tf.Session(config=tf_config) as sess:
        model=cyclegan(sess,FLAGS)
        model.train()
        #model.train() if FLAGS.phase == 'train' \
        #    else model.test()
#URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip

if __name__ == '__main__':
    tf.app.run()