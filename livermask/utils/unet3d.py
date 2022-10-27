import chainer
import chainer.functions as F
import chainer.links as L


class UNet3D(chainer.Chain):
    def __init__(self, num_of_label):
        w = chainer.initializers.HeNormal()
        super(UNet3D, self).__init__()
        with self.init_scope():
            # encoder pass
            self.ce0 = L.ConvolutionND(ndim=3, in_channels=1, out_channels=16, ksize=3, pad=1, initialW=w)
            self.bne0 = L.BatchNormalization(16)
            self.ce1 = L.ConvolutionND(ndim=3, in_channels=16, out_channels=32, ksize=3, pad=1, initialW=w)
            self.bne1 = L.BatchNormalization(32)

            self.ce2 = L.ConvolutionND(ndim=3, in_channels=32, out_channels=32, ksize=3, pad=1, initialW=w)
            self.bne2 = L.BatchNormalization(32)
            self.ce3 = L.ConvolutionND(ndim=3, in_channels=32, out_channels=64, ksize=3, pad=1, initialW=w)
            self.bne3 = L.BatchNormalization(64)

            self.ce4 = L.ConvolutionND(ndim=3, in_channels=64, out_channels=64, ksize=3, pad=1, initialW=w)
            self.bne4 = L.BatchNormalization(64)

            # decoder pass
            self.cd4 = L.ConvolutionND(ndim=3, in_channels=64, out_channels=128, ksize=3, pad=1, initialW=w)
            self.bnd4 = L.BatchNormalization(128)
            self.deconv2 = L.DeconvolutionND(ndim=3, in_channels=128, out_channels=128, ksize=2, stride=2, initialW=w,
                                             nobias=True)

            self.cd3 = L.ConvolutionND(ndim=3, in_channels=64 + 128, out_channels=64, ksize=3, pad=1, initialW=w)
            self.bnd3 = L.BatchNormalization(64)
            self.cd2 = L.ConvolutionND(ndim=3, in_channels=64, out_channels=64, ksize=3, pad=1, initialW=w)
            self.bnd2 = L.BatchNormalization(64)
            self.deconv1 = L.DeconvolutionND(ndim=3, in_channels=64, out_channels=64, ksize=2, stride=2, initialW=w,
                                             nobias=True)

            self.cd1 = L.ConvolutionND(ndim=3, in_channels=32 + 64, out_channels=32, ksize=3, pad=1, initialW=w)
            self.bnd1 = L.BatchNormalization(32)
            self.cd0 = L.ConvolutionND(ndim=3, in_channels=32, out_channels=32, ksize=3, pad=1, initialW=w)
            self.bnd0 = L.BatchNormalization(32)
            self.lcl = L.ConvolutionND(ndim=3, in_channels=32, out_channels=num_of_label, ksize=1, pad=0, initialW=w)

    def __call__(self, x):
        # encoder pass
        e0 = F.relu(self.bne0(self.ce0(x)))
        e1 = F.relu(self.bne1(self.ce1(e0)))
        del e0
        e2 = F.relu(self.bne2(self.ce2(F.max_pooling_nd(e1, ksize=2, stride=2))))
        e3 = F.relu(self.bne3(self.ce3(e2)))
        del e2
        e4 = F.relu(self.bne4(self.ce4(F.max_pooling_nd(e3, ksize=2, stride=2))))

        # decoder pass
        d4 = F.relu(self.bnd4(self.cd4(e4)))
        del e4
        d3 = F.relu(self.bnd3(self.cd3(F.concat([self.deconv2(d4), e3]))))
        del d4, e3
        d2 = F.relu(self.bnd2(self.cd2(d3)))
        del d3
        d1 = F.relu(self.bnd1(self.cd1(F.concat([self.deconv1(d2), e1]))))
        del d2, e1
        d0 = F.relu(self.bnd0(self.cd0(d1)))
        del d1
        lcl = F.softmax(self.lcl(d0), axis=1)

        return lcl  # (batchsize, ch, z, y, x)

    def cropping(self, input, ref):
        '''
        * @param input encoder feature map
        * @param ref decoder feature map
        '''
        edgez = (input.shape[2] - ref.shape[2]) / 2
        edgey = (input.shape[3] - ref.shape[3]) / 2
        edgex = (input.shape[4] - ref.shape[4]) / 2
        edgez = int(edgex)
        edgey = int(edgey)
        edgex = int(edgez)

        X = F.split_axis(input, (edgex, int(input.shape[4] - edgex)), axis=4)
        X = X[1]
        X = F.split_axis(X, (edgey, int(X.shape[3] - edgey)), axis=3)
        X = X[1]
        X = F.split_axis(X, (edgez, int(X.shape[2] - edgez)), axis=2)
        X = X[1]
        return X
