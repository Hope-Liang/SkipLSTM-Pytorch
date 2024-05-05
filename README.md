# SkipLSTM-Pytorch
A PyTorch implementation of SkipLSTM for Pytorch 2.3

This implementation is modified from [skiprnn_pytorch](https://github.com/gitabcworld/skiprnn_pytorch), and corresponds to the paper [Skip RNN: Learning to Skip State Updates in Recurrent Neural Networks](https://arxiv.org/abs/1708.06834).

This works well with Pytorch latest version 2.3 [torch.nn.RNNCell](https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html).

Limitations: Only single-layer SkipLSTM cell is supported.
