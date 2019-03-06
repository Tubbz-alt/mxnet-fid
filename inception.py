import mxnet as mx
import gluoncv as gcv


class InceptionV3(mx.gluon.Block):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 2

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        2048: 2  # Final average pooling features
    }

    def __init__(self,
                 ctx,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True):
        """Build pretrained InceptionV3

        Parameters
        ----------
        ctx : mxnet context (gpu or cpu)
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradient. Possibly useful
            for finetuning the network
        """
        super(InceptionV3, self).__init__()

        self.ctx = ctx
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 2, \
            'Last possible output block index is 2'

        self.blocks = list()

        # based on this discussion: https://discuss.mxnet.io/t/gluon-pretrained-model-layer-access-and-usage/772/2
        inception = gcv.model_zoo.get_model('inceptionv3', pretrained=True)
        inception.collect_params().reset_ctx(self.ctx)
        inception.hybridize()
        inception(mx.nd.ones((1, 3, 512, 512), ctx=self.ctx))
        inception.export('inception')
        sym = mx.sym.load('inception-symbol.json')

        # Block 0: input to maxpool1
        block0_sym = sym.get_internals()['inception30_pool0_fwd_output']
        block0 = mx.gluon.nn.SymbolBlock(outputs=block0_sym, inputs=mx.sym.var('data'))
        block0.collect_params().load('inception-0000.params', ctx=self.ctx, ignore_extra=True)
        self.blocks.append(block0)

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block < 1:
            return

        block1_sym = sym.get_internals()['inception30_pool1_fwd_output']
        block1 = mx.gluon.nn.SymbolBlock(outputs=block1_sym, inputs=mx.sym.var('data'))
        block1.collect_params().load('inception-0000.params', ctx=self.ctx, ignore_extra=True)
        self.blocks.append(block1)

        # Block 2: aux classifier to final avgpool
        if self.last_needed_block < 2:
            return

        block2_sym = sym.get_internals()['inception30_pool2_fwd_output']
        block2 = mx.gluon.nn.SymbolBlock(outputs=block2_sym, inputs=mx.sym.var('data'))
        block2.collect_params().load('inception-0000.params', ctx=self.ctx, ignore_extra=True)
        self.blocks.append(block2)


    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:

            grids = mx.nd.GridGenerator(data=mx.nd.array([[1,0,0,0,1,0]], ctx=self.ctx), transform_type='affine',
                                    target_shape=(299, 299))
            x = mx.nd.BilinearSampler(x, mx.nd.tile(grids, (x.shape[0],1,1,1)))

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        x_init = x
        for idx, block in enumerate(self.blocks):
            x = block(x_init)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp