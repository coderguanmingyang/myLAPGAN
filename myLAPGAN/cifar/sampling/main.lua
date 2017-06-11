--
-- Created by IntelliJ IDEA.
-- User: PC
-- Date: 2017/5/10
-- Time: 19:22
-- To change this template use File | Settings | File Templates.
--
require 'torch'
require 'cunn'
require 'nngraph'
require 'optim'
require 'image'
require 'pl'
require 'paths'
image_utils = require 'utils.image'
get3layer = require'sampling.getSamples3layers'
get2layer = require'sampling.getSamples2layers'
require 'layers.SpatialConvolutionUpsample'
adversarial8 = require 'train.double_conditional_adversarial-8'
require 'datasets.coarse_to_fine_cifar10-8'

batchsize = 24
noiseDim8  = { 4, 8, 8 }
noiseDim16 = { 1, 16, 16 }
noiseDim32 = { 1, 32, 32 }

--torch.manualSeed(8)
 --noise_inputs8 = torch.Tensor(1, 2, 2, 2) --128*3*8*8
 --noise_inputs8:uniform()
 --print(noise_inputs8)
--print('\n unsamlple:')
--get3layer.genImage('/home/guanmingyang/myproject/G-LAPGAN/cifar/s1_k2_sc8.net', '/home/guanmingyang/myproject/G-LAPGAN/cifar/un_s1_k1_50_sc16.net','/home/guanmingyang/myproject/G-LAPGAN/cifar/un_s1_k1_15_sc32.net')
--print('\n sample:')
get3layer.genImage('/home/guanmingyang/myproject/G-LAPGAN/cifar/s1_k2_sc8.net', '/home/guanmingyang/myproject/G-LAPGAN/cifar/up_s1_k1_50_sc16.net','/home/guanmingyang/myproject/G-LAPGAN/cifar/up_s3_k1_15_sc32.net')
--[[
cifar_scale8.init(8)
  -- create training set and normalize
trainData8 = cifar_scale8.loadTrainSet(1, 45000) -- start = 1 stop = 45000
mean8, std8 = image_utils.normalize(trainData8.data)
trainData8:makeFine()

--get2layer.genImage(trainData8,'/home/guanmingyang/myproject/G-LAPGAN/cifar/un_s1_k1_50_sc16.net','/home/guanmingyang/myproject/G-LAPGAN/cifar/un_s1_k1_15_sc32.net')
get2layer.genImage(trainData8,'/home/guanmingyang/myproject/G-LAPGAN/cifar/un_s1_k1_50_sc16.net','/home/guanmingyang/myproject/G-LAPGAN/cifar/s1_k2_sc32.net')
]]--