import tensorflow as tf
import model
import os,time
import loader
import numpy as np
class cyclegan(object):
    def __init__(self,sess,flags):

        self.step=tf.Variable(0, trainable=False)
        self.sess=sess
        self.flags=flags
        self.image_size=flags.fine_size
        self.L1_lambda=flags.L1_lambda

        self.generator=model.generator
        self.discriminator=model.discriminator
        self.lossfunc=model.MSE_loss

        self._build_model()
        self.pool=loader.ImagePool(self.flags.max_size)


    def _build_model(self):
        self.real_data=tf.placeholder(tf.float32,[None, self.image_size, self.image_size,3+3],name='real_A_and_B_images')

        self.real_A = self.real_data[:, :, :, :3]
        self.real_B = self.real_data[:, :, :, 3:3+3]
        self.G_A= self.generator(self.real_A, reuse= False, name="generatorA2B")
        self.F_G_A= self.generator(self.G_A, reuse= False, name="generatorB2A")
        self.F_B=self.generator(self.real_B,reuse=True,name="generatorB2A")
        self.G_F_B=self.generator(self.F_B,reuse=True,name="generatorA2B")

        self.DB_fake=self.discriminator(self.G_A, reuse=False, name="discriminatorB")
        self.DA_fake = self.discriminator(self.F_B, reuse=False, name="discriminatorA")


        self.g_loss_a2b=self.lossfunc(self.DB_fake, tf.ones_like(self.DB_fake)) \
            + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_A- self.F_G_A)) \
            + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B- self.G_F_B))

        self.g_loss_b2a=self.lossfunc(self.DA_fake, tf.ones_like(self.DA_fake)) \
            + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_A-self.F_G_A)) \
            + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B-self.G_F_B))

        self.g_loss= self.lossfunc(self.DB_fake, tf.ones_like(self.DB_fake)) \
                +self.lossfunc(self.DA_fake, tf.ones_like(self.DA_fake)) \
                + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_A- self.F_G_A)) \
                + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B- self.G_F_B))

        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             3], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             3], name='fake_B_sample')
        self.DB_real = self.discriminator(self.real_B, reuse=True, name="discriminatorB")
        self.DA_real = self.discriminator(self.real_A, reuse=True, name="discriminatorA")
        self.DB_fake_sample = self.discriminator(self.fake_B_sample,reuse=True, name="discriminatorB")
        self.DA_fake_sample = self.discriminator(self.fake_A_sample, reuse=True, name="discriminatorA")

        self.db_loss=0.5*(self.lossfunc(self.DB_real, tf.ones_like(self.DB_real))+self.lossfunc(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample)))
        self.da_loss=0.5*(self.lossfunc(self.DA_real, tf.ones_like(self.DA_real))+self.lossfunc(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample)))
        self.d_loss = self.da_loss + self.db_loss

        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum])
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.d_sum = tf.summary.merge([self.da_loss_sum, self.db_loss_sum, self.d_loss_sum])

        self.test_A = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      3], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      3], name='test_B')
        self.testG_A = self.generator(self.test_A, reuse=True, name="generatorA2B")
        self.testF_B = self.generator(self.test_B, reuse=True, name="generatorB2A")

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]

        for var in t_vars: print(var.name)






    def train(self):
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=self.flags.beta1) \
            .minimize(self.d_loss,var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=self.flags.beta1) \
            .minimize(self.g_loss,global_step=self.step,var_list=self.g_vars)


        init = tf.global_variables_initializer()
        self.sess.run(init)
        global_step=0
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=1)

        A2B=False
        if A2B:
            sample_files = loader.load_train(self.flags.Atpath)
            out_var, in_var = (self.testG_A, self.test_A)
        else:
            sample_files = loader.load_train(self.flags.Btpath)
            out_var, in_var = (self.testF_B, self.test_B)


        if (self.flags.is_continue_train):

            ckpt = tf.train.get_checkpoint_state("./parameters/")
            if ckpt and ckpt.model_checkpoint_path:
                print('successfully load')
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        for epoch in range(self.flags.epoch):

            dataA=loader.load_train(self.flags.Apath)
            dataB=loader.load_train(self.flags.Bpath)
            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
            batch_idxs = min(len(dataA), len(dataB))
            lr = self.flags.lr if epoch < self.flags.epoch_step else self.flags.lr * (self.flags.epoch - epoch) / (self.flags.epoch - self.flags.epoch_step)
            for idx in range(batch_idxs):
                start_time = time.time()

                batch_images=[loader.load_train_data([dataA[idx],dataB[idx]],load_size=286, fine_size=256,is_testing=False)]
                batch_images=np.array(batch_images).astype(np.float32)

                fake_A, fake_B, _, summary_str,global_step = self.sess.run(
                    [self.F_B, self.G_A, self.g_optim, self.g_sum,self.step],
                    feed_dict={self.real_data: batch_images, self.lr: lr})
                self.writer.add_summary(summary_str, global_step)
                [fake_A, fake_B] = self.pool([fake_A, fake_B])

                # Update D network
                _, summary_str = self.sess.run(
                    [self.d_optim, self.d_sum],
                    feed_dict={self.real_data: batch_images,
                               self.fake_A_sample: fake_A,
                               self.fake_B_sample: fake_B,
                               self.lr: lr})
                self.writer.add_summary(summary_str, global_step)
                print(("Epoch: [%2d] [%4d/%4d] [%4d] Cost time: %4.4f" % (
                    epoch, idx, batch_idxs,global_step, time.time() - start_time)))
                if global_step%500000==0 and global_step>10:
                    sample_dir='sample'
                    if not os.path.isdir('./'+sample_dir):
                        os.mkdir('./'+sample_dir)
                    loader.save_images(fake_A,[1,1],'./{}/A_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
                    loader.save_images(fake_B,[1,1],'./{}/B_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
                if global_step%50==0 and global_step>10:
                    np.random.shuffle(sample_files)
                    sample_file=sample_files[0]
                    print('Processing image: ' + sample_file)
                    out_dir = './test_fack'
                    if not os.path.isdir(out_dir):
                        os.mkdir(out_dir)
                    sample_image = [loader.load_test_data(sample_file)]
                    sample_image = np.array(sample_image).astype(np.float32)
                    fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
                    loader.save_images(sample_image, [1, 1], os.path.join(out_dir, str(global_step)+os.path.splitext(os.path.basename(sample_file))[0] + '_r.jpg'))
                    loader.save_images(fake_img, [1, 1], os.path.join(out_dir, str(global_step)+os.path.splitext(os.path.basename(sample_file))[0] + '_f.jpg'))

            self.saver.save(self.sess, "./parameters/model"+str(epoch)+".ckpt",global_step=global_step)


    def test(self):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state("./parameters/")
        if ckpt and ckpt.model_checkpoint_path:
            print('successfully load')
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('failed to load')
            return
        A2B=True
        if A2B:
            sample_files = loader.load_train(self.flags.Atpath)
            out_var, in_var = (self.testG_A, self.test_A)
        else:
            sample_files = loader.load_train(self.flags.Btpath)
            out_var, in_var = (self.testF_B, self.test_B)

        for sample_file in sample_files:
            print('Processing image: ' + sample_file)

            out_dir = './test_fack'
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)

            sample_image = [loader.load_test_data(sample_file)]
            sample_image = np.array(sample_image).astype(np.float32)
            fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})

            loader.save_images(sample_image, [1, 1],  os.path.join(out_dir, os.path.splitext(os.path.basename(sample_file))[0] + '_r.jpg'))
            loader.save_images(fake_img, [1, 1],  os.path.join(out_dir, os.path.splitext(os.path.basename(sample_file))[0] + '_f.jpg'))

