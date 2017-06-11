require 'torch'
require 'cunn'
require 'nngraph'
require 'optim'
require 'image'
require 'pl'
require 'paths'
require 'datasets.coarse_to_fine_cifar10-32'
require 'datasets.coarse_to_fine_cifar10-16'
require 'datasets.coarse_to_fine_cifar10-8'
adversarial32 = require 'train.double_conditional_adversarial-32'
adversarial16 = require 'train.double_conditional_adversarial-16'
adversarial8 = require 'train.double_conditional_adversarial-8'
image_utils = require 'utils.image'
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
require 'layers.SpatialConvolutionUpsample'
getsamples = require 'sampling.getSamples2layers'

----------------------------------------------------------------------
-- parse command-line options
opt = lapp[[
  --save8           (default "logs8")      subdirectory to save logs
  --save16           (default "logs16")      subdirectory to save logs
  --save32           (default "logs32")      subdirectory to save logs
  --saveFreq         (default 30)           save every saveFreq epochs
  --network8         (default "")          reload pretrained network of scale 8
  --network16        (default "")          reload pretrained network of scale 16
  --network32        (default "")          reload pretrained network of scale 32
  -p,--plot                                plot while training
  -r,--learningRate  (default 0.02)        learning rate
  -b,--batchSize     (default 128)         batch size
  -m,--momentum      (default 0)           momentum
  --coefL1           (default 0)           L1 penalty on the weights
  --coefL2           (default 0)           L2 penalty on the weights
  -t,--threads       (default 4)           number of threads
  -g,--gpu           (default -1)          gpu to run on (default cpu)
  --K                (default 1)           number of iterations to optimize D for
  -w, --window       (default 3)           windsow id of sample image
  --hidden_G8        (default 64)         number of channels in hidden layers of G in 8-scale
  --hidden_D8        (default 64)         number of channels in hidden layers of D in 8-scale
  --hidden_G16       (default 64)         number of channels in hidden layers of G in 16-scale
  --hidden_D16       (default 64)         number of channels in hidden layers of D in 16-scale
  --hidden_G32       (default 64)         number of channels in hidden layers of G in 32-scale
  --hidden_D32       (default 64)         number of channels in hidden layers of D in 32-scale
  --ifwbupsampleD1     (default 0)
  --ifwbupsampleG1     (default 0)
  --ifwbupsampleD2     (default 0)
  --ifwbupsampleG2     (default 0)
  --mode              (default 1)
  --seed              (default 1)
]]

if opt.gpu < 0 or opt.gpu > 3 then opt.gpu = false end

if opt.ifwbupsample == 1  then
  opt.save16 = paths.concat('wbupsample/',opt.save16)
  opt.save32 = paths.concat('wbupsample/',opt.save32)
end

print(opt)

-- fix seed
torch.manualSeed(opt.seed)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())
-- GPU model
if opt.gpu then
  --cutorch.setDevice(opt.gpu + 1) --Mark
  cutorch.setDevice(opt.gpu )
  print('<gpu> using device ' .. opt.gpu)
  torch.setdefaulttensortype('torch.CudaTensor')
else
  torch.setdefaulttensortype('torch.FloatTensor')
end

-- classes
classes = {'0','1'}
cifar_classes =  {'airplane', 'automobile', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

------------------------------------------------------------------------
--network
--************scale 8 network
if opt.network8 == '' then
  --------------------------------------------------------
  -- define D network to train
  local nplanes = opt.hidden_D8
  d_fine_sc8  = nn.Identity()()
  d_c1_sc8  = nn.Identity()()
  d_class1_sc8 = nn.Linear(10, 8*8)(d_c1_sc8)
  d_class2_sc8= nn.Reshape(1, 8, 8)(nn.ReLU()(d_class1_sc8))
  d1_sc8 = nn.JoinTable(2, 2)({d_fine_sc8, d_class2_sc8})
  d2_sc8 = nn.SpatialConvolution(3+1, nplanes, 3, 3)(d1_sc8)
  d3_sc8 = nn.SpatialConvolution(nplanes, nplanes, 3, 3, 1, 1)(nn.ReLU()(d2_sc8))
  --local sz =math.floor( ( (8 - 5 + 1) - 5) / 2 + 1)  --floor 求比这个数小的最大整数 12
  d4_sc8 = nn.Reshape(nplanes*4*4)(d3_sc8)
  d5_sc8 = nn.Linear(nplanes*4*4, 1)(nn.Dropout()(nn.ReLU()(d4_sc8)))
  d6_sc8 = nn.Sigmoid()(d5_sc8)
  model_D8 = nn.gModule({d_fine_sc8, d_c1_sc8}, {d6_sc8})

  -------------------------------------- --------------------------------
  -- define G network to train
  local nplanes = opt.hidden_G8
  --用来存放输入数据：identity
  g_n_sc8  = nn.Identity()() -- noise (shaped as coarse map)
  g_c_sc8 = nn.Identity()() -- class vector

  g_class1_sc8 = nn.Linear(10, 8*8)(g_c_sc8) -- 10 inputs -> fine size * fine size outputs (default 16)
  g_class2_sc8 = nn.Reshape(1, 8, 8)(nn.ReLU()(g_class1_sc8)) --convert class vector into map
  -- 按列（第2维）
  g1_sc8 = nn.JoinTable(2, 2)({g_n_sc8, g_class2_sc8}) -- combine maps into 4 channels
  g2_sc8 = nn.SpatialConvolutionUpsample(5, nplanes, 3, 3, 1)(g1_sc8)   --filter nplanes*3*3
  g3_sc8 = nn.SpatialConvolutionUpsample(nplanes, nplanes, 3, 3, 1)(nn.ReLU()(g2_sc8))
  g4_sc8 = nn.SpatialConvolutionUpsample(nplanes, 3, 3, 3, 1)(nn.ReLU()(g3_sc8))
  model_G8 = nn.gModule({g_n_sc8, g_c_sc8}, {g4_sc8}) --生成一个有向无环图

else
  print('<trainer> reloading previously trained network: ' .. opt.network8)
  tmp = torch.load(opt.network8)
  model_D8 = tmp.D
  model_G8 = tmp.G
end
--------------------------------------------------------------------------
--************scale 16 network*******************************
if opt.network16 == '' then
  --------------------------------------------------------
  -- define D network to train
  local nplanes = opt.hidden_D16
  x_diff16  = nn.Identity()()
  x_c1_sc16 = nn.Identity()()
  x_c2_sc16 = nn.Identity()()
  d1_sc16 = nn.CAddTable()({x_diff16, x_c2_sc16})
  c1_sc16 = nn.Linear(condDim1_sc16, condDim2_sc16[2]*condDim2_sc16[3])(x_c1_sc16)
  c2_sc16 = nn.Reshape(1, condDim2_sc16[2], condDim2_sc16[3])(nn.ReLU()(c1_sc16))
  d2_sc16 = nn.JoinTable(2, 2)({d1_sc16, c2_sc16})
  d3_sc16 = nn.SpatialConvolution(3+1, nplanes, 5, 5)(d2_sc16)
  --d4_sc16 = nn.SpatialConvolution(nplanes, nplanes, 5, 5, 2, 2)(nn.ReLU()(d3_sc16))
  --local sz =math.floor( ( (16 - 5 + 1) - 5) / 2 + 1)  --floor 求比这个数小的最大整数 12
  d4_sc16 = nn.SpatialConvolution(nplanes, nplanes, 5, 5, 2, 2)(nn.ReLU()(d3_sc16))
  local sz = 4
  d5_sc16 = nn.Reshape(nplanes*sz*sz)(d4_sc16)
  d6_sc16 = nn.Linear(nplanes*sz*sz, 1)(nn.Dropout()(nn.ReLU()(d5_sc16)))
  d7_sc16 = nn.Sigmoid()(d6_sc16)
  model_D16 = nn.gModule({x_diff16, x_c1_sc16, x_c2_sc16}, {d7_sc16})

  ----------------------------------------------------------------------
  -- define G network to train
  local nplanes = opt.hidden_G16
  --用来存放输入数据：identity
  x_n_sc16  = nn.Identity()() -- noise (shaped as coarse map)
  g_c1_sc16 = nn.Identity()() -- class vector
  g_c2_sc16 = nn.Identity()() -- coarse map

  class1_sc16 = nn.Linear(condDim1_sc16, condDim2_sc16[2]*condDim2_sc16[3])(g_c1_sc16) -- 10 inputs -> fine size * fine size outputs (default 16)
  class2_sc16 = nn.Reshape(1, condDim2_sc16[2], condDim2_sc16[3])(nn.ReLU()(class1_sc16)) --convert class vector into map
  -- 按列（第2维）
  g1_sc16 = nn.JoinTable(2, 2)({x_n_sc16, class2_sc16, g_c2_sc16}) -- combine maps into 5 channels
  g2_sc16 = nn.SpatialConvolutionUpsample(5, nplanes, 5, 5, 1)(g1_sc16)   --filter nplanes*3*3
  g3_sc16 = nn.SpatialConvolutionUpsample(nplanes, nplanes, 5, 5, 1)(nn.ReLU()(g2_sc16))
  g4_sc16 = nn.SpatialConvolutionUpsample(nplanes, 3, 5, 5, 1)(nn.ReLU()(g3_sc16))
  model_G16 = nn.gModule({x_n_sc16, g_c1_sc16, g_c2_sc16}, {g4_sc16}) --生成一个有向无环图

else
  print('<trainer> reloading previously trained network: ' .. opt.network16)
  tmp = torch.load(opt.network16)
  model_D16 = tmp.D
  model_G16 = tmp.G
end

--************************
--********scale 32 network
if opt.network32 == '' then
  ----------------------------------------------------------------------
  -- define D network to train
 local nplanes = opt.hidden_D32
  x_diff32  = nn.Identity()()
  x_c1_sc32 = nn.Identity()()
  x_c2_sc32 = nn.Identity()()
  d1_sc32 = nn.CAddTable()({x_diff32, x_c2_sc32})
  c1_sc32 = nn.Linear(condDim1_sc32, condDim2_sc32[2]*condDim2_sc32[3])(x_c1_sc32)
  c2_sc32 = nn.Reshape(1, condDim2_sc32[2], condDim2_sc32[3])(nn.ReLU()(c1_sc32))
  d2_sc32 = nn.JoinTable(2, 2)({d1_sc32, c2_sc32})
  d3_sc32 = nn.SpatialConvolution(3+1, nplanes, 7, 7)(d2_sc32)
  d4_sc32 = nn.SpatialConvolution(nplanes, nplanes, 7, 7, 2, 2)(nn.ReLU()(d3_sc32))
  --local sz =math.floor( ( (32 - 5 + 1) - 5) / 2 + 1)  --floor 求比这个数小的最大整数 12
  local sz = 10
  d5_sc32 = nn.Reshape(nplanes*sz*sz)(d4_sc32)
  d6_sc32 = nn.Linear(nplanes*sz*sz, 1)(nn.Dropout()(nn.ReLU()(d5_sc32)))
  d7_sc32 = nn.Sigmoid()(d6_sc32)
  model_D32 = nn.gModule({x_diff32, x_c1_sc32, x_c2_sc32}, {d7_sc32})

  ----------------------------------------------------------------------
  -- define G network to train
  local nplanes = opt.hidden_G32
  --用来存放输入数据：identity
  x_n_sc32 = nn.Identity()() -- noise (shaped as coarse map)
  g_c1_sc32 = nn.Identity()() -- class vector
  g_c2_sc32 = nn.Identity()() -- coarse map

  class1_sc32 = nn.Linear(condDim1_sc32, condDim2_sc32[2]*condDim2_sc32[3])(g_c1_sc32) -- 10 inputs -> fine size * fine size outputs (default 32)
  class2_sc32 = nn.Reshape(1, condDim2_sc32[2], condDim2_sc32[3])(nn.ReLU()(class1_sc32)) --convert class vector into map
  -- 按列（第2维）
  g1_sc32 = nn.JoinTable(2, 2)({x_n_sc32, class2_sc32, g_c2_sc32}) -- combine maps into 5 channels
  g2_sc32 = nn.SpatialConvolutionUpsample(5, nplanes, 7, 7, 1)(g1_sc32)   --filter nplanes*7*7
  g3_sc32 = nn.SpatialConvolutionUpsample(nplanes, nplanes, 7, 7, 1)(nn.ReLU()(g2_sc32))
  g4_sc32 = nn.SpatialConvolutionUpsample(nplanes, 3, 7, 7, 1)(nn.ReLU()(g3_sc32))
  model_G32 = nn.gModule({x_n_sc32, g_c1_sc32, g_c2_sc32}, {g4_sc32}) --生成一个有向无环图

else
  print('<trainer> reloading previously trained network: ' .. opt.network32)
  tmp = torch.load(opt.network32)
  model_D32 = tmp.D
  model_G32 = tmp.G
end

-- Binary Cross Entropy
-- loss function: negative log-likelihood
-- Creates a criterion that measures the Binary Cross Entropy between the target and the output:
-- crossentropy(t,o) = -(t * log(o) + (1 - t) * log(1 - o))
criterion32 = nn.BCECriterion()
criterion16 = nn.BCECriterion()
criterion8  = nn.BCECriterion()

-- retrieve parameters and gradients
parameters_D8,gradParameters_D8 = model_D8:getParameters()
parameters_G8,gradParameters_G8 = model_G8:getParameters()

parameters_D16,gradParameters_D16 = model_D16:getParameters()
parameters_G16,gradParameters_G16 = model_G16:getParameters()

parameters_D32,gradParameters_D32 = model_D32:getParameters()
parameters_G32,gradParameters_G32 = model_G32:getParameters()

-- print networks
pa_D8 = model_D8:parameters()
pa_G8 = model_G8:parameters()

pa_D16 = model_D16:parameters()
pa_G16 = model_G16:parameters()

pa_D32 = model_D32:parameters()
pa_G32 = model_G32:parameters()

print('\n'..'***********scale8')
print('\n'..'**********[Discriminator network]**********')
print('[Parameters:]')
print(pa_D8)
print('size: ' .. model_D8:size())
for i=1,model_D8:size() do
  print(i,model_D8:get(i))
end
print('\n'..'**************[Generator network]**********')
print('[Parameters:]')
print(pa_G8)
print('size: ' .. model_G8:size())
for i=1,model_G8:size() do
  print(i,model_G8:get(i))
end

print('\n'..'***********scale16')
print('\n'..'**********[Discriminator network]**********')
print('[Parameters:]')
print(pa_D16)
print('size: ' .. model_D16:size())
for i=1,model_D16:size() do
  print(i,model_D16:get(i))
end
print('\n'..'**************[Generator network]**********')
print('[Parameters:]')
print(pa_G16)
print('size: ' .. model_G16:size())
for i=1,model_G16:size() do
  print(i,model_G16:get(i))
end

print('\n'..'***********scale32')
print('\n'..'**********[Discriminator network]**********')
print('[Parameters:]')
print(pa_D32)
print('size: ' .. model_D32:size())
for i=1,model_D32:size() do
  print(i,model_D32:get(i))
end
print('\n'..'**************[Generator network]**********')
print('[Parameters:]')
print(pa_G32)
print('size: ' .. model_G32:size())
for i=1,model_G32:size() do
  print(i,model_G32:get(i))
end

print('\nscale 16:')
local nparams = 0
for i=1,#model_D16.forwardnodes do
  if model_D16.forwardnodes[i].data.module ~= nil and model_D16.forwardnodes[i].data.module.weight ~= nil then
    nparams = nparams + model_D16.forwardnodes[i].data.module.weight:nElement()
  end
end
print('Number of free parameters in D: ' .. nparams)

local nparams = 0
for i=1,#model_G16.forwardnodes do
  if model_G16.forwardnodes[i].data.module ~= nil and model_G16.forwardnodes[i].data.module.weight ~= nil then
    nparams = nparams + model_G16.forwardnodes[i].data.module.weight:nElement()
  end
end
print('Number of free parameters in G: ' .. nparams .. '\n')

print('\nscale 32:')
local nparams = 0
for i=1,#model_D32.forwardnodes do
  if model_D32.forwardnodes[i].data.module ~= nil and model_D32.forwardnodes[i].data.module.weight ~= nil then
    nparams = nparams + model_D32.forwardnodes[i].data.module.weight:nElement()
  end
end
print('Number of free parameters in D: ' .. nparams)

local nparams = 0
for i=1,#model_G32.forwardnodes do
  if model_G32.forwardnodes[i].data.module ~= nil and model_G32.forwardnodes[i].data.module.weight ~= nil then
    nparams = nparams + model_G32.forwardnodes[i].data.module.weight:nElement()
  end
end
print('Number of free parameters in G: ' .. nparams .. '\n')

----------------------------------------------------------------------
-- get/create dataset
ntrain = 45000
nval = 5000
-- **************scale 8
if opt.mode == 8 or opt.mode == 1  then
  cifar_scale8.init(8)
  -- create training set and normalize
  trainData8 = cifar_scale8.loadTrainSet(1, ntrain) -- start = 1 stop = 45000
  mean8, std8 = image_utils.normalize(trainData8.data)
  trainData8:makeFine()

  -- create validation set and normalize
  valData8 = cifar_scale8.loadTrainSet(ntrain+1, ntrain+nval)
  image_utils.normalize(valData8.data, mean8, std8)
  valData8:makeFine()

  trainLogger8 = optim.Logger(paths.concat(opt.save8, 'train8.log'))
  testLogger8 = optim.Logger(paths.concat(opt.save8, 'test8.log'))
  avdistanceLogger8 = optim.Logger(paths.concat(opt.save8, 'avdistance8.log'))
  maxdistanceLogger8 = optim.Logger(paths.concat(opt.save8, 'maxdistance8.log'))
  mindistanceLogger8 = optim.Logger(paths.concat(opt.save8, 'mindistance8.log'))
end

-- **************scale 16
if opt.mode == 16 or opt.mode == 1   then

  cifar_scale16.init(16, 8)
  -- create training set and normalize
  trainData16 = cifar_scale16.loadTrainSet(1, ntrain) -- start = 1 stop = 45000
  mean16, std16 = image_utils.normalize(trainData16.data)
  trainData16:makeFine()
  trainData16:makeCoarse()
  trainData16:makeDiff()
  -- create validation set and normalize
  valData16 = cifar_scale16.loadTrainSet(ntrain+1, ntrain+nval)
  image_utils.normalize(valData16.data, mean16, std16)
  valData16:makeFine()
  valData16:makeCoarse()
  valData16:makeDiff()

  trainLogger16 = optim.Logger(paths.concat(opt.save16, 'train16.log'))
  testLogger16 = optim.Logger(paths.concat(opt.save16, 'test16.log'))
  avdistanceLogger16 = optim.Logger(paths.concat(opt.save16, 'avdistance16.log'))
  maxdistanceLogger16 = optim.Logger(paths.concat(opt.save16, 'maxdistance16.log'))
  mindistanceLogger16 = optim.Logger(paths.concat(opt.save16, 'mindistance16.log'))
end
--**************scale 32
if opt.mode == 32 or opt.mode == 1  then

  cifar_scale32.init(32, 16)
-- create training set and normalize
  trainData32 = cifar_scale32.loadTrainSet(1, ntrain) -- start = 1 stop = 45000
  mean32, std32 = image_utils.normalize(trainData32.data)
  trainData32:makeFine()
  trainData32:makeCoarse()
  trainData32:makeDiff()

  -- create validation set and normalize
  valData32 = cifar_scale32.loadTrainSet(ntrain+1, ntrain+nval)
  image_utils.normalize(valData32.data, mean32, std32)
  valData32:makeFine()
  valData32:makeCoarse()
  valData32:makeDiff()

  trainLogger32 = optim.Logger(paths.concat(opt.save32, 'train32.log'))
  testLogger32 = optim.Logger(paths.concat(opt.save32, 'test32.log'))
  avdistanceLogger32 = optim.Logger(paths.concat(opt.save32, 'avdistance32.log'))
  maxdistanceLogger32 = optim.Logger(paths.concat(opt.save32, 'maxdistance32.log'))
  mindistanceLogger32 = optim.Logger(paths.concat(opt.save32, 'mindistance32.log'))
end
------------------------------------------------------

-- this matrix records the current confusion across classes
confusion32 = optim.ConfusionMatrix(classes) ---- 定义混淆矩阵用于评价模型性能，后续计算正确率，召回率等
confusion16 = optim.ConfusionMatrix(classes)
confusion8 = optim.ConfusionMatrix(classes)

-- log results to files
--logger16 = optim.Logger(paths.concat(opt.save16, 'logger16.log'))
--logger16:setNames{'% mean class accuracy (train set)','% mean class accuracy (test set)'}

--logger32 = optim.Logger(paths.concat(opt.save16, 'logger32.log'))
--logger32:setNames{'% mean class accuracy (train set)','% mean class accuracy (test set)'}

if opt.gpu then
  print('Copy model to gpu')
  model_D8:cuda()
  model_G8:cuda()
  model_D16:cuda()
  model_G16:cuda()
  model_D32:cuda()
  model_G32:cuda()
end
-- Training parameters config
sgdState_D = {
  learningRate = opt.learningRate,
  momentum = opt.momentum
}

sgdState_G = {
  learningRate = opt.learningRate,
  momentum = opt.momentum
}
sgdState_D.momentum = 0.0008
sgdState_D.learningRate = 0.02
sgdState_G.momentum = 0.0008
sgdState_G.learningRate = 0.02
--torch.load('/home/guanmingyang/myproject/LAPGAN/cifar/wbupsample/20000-2000-50-31/logs32/conditional_adversarial_sc32.net')
--print('\nsamlple:')
--getsamples.genImage(valData16,'/home/guanmingyang/myproject/LAPGAN/cifar/wbupsample/20000-2000-50-31/logs16/conditional_adversarial_sc16.net','/home/guanmingyang/myproject/LAPGAN/cifar/wbupsample/20000-2000-50-31/logs32/conditional_adversarial_sc32.net')
--print('\nunsample:')
--getsamples.genImage(valData16,'/home/guanmingyang/myproject/LAPGAN/cifar/wbupsample/20000-2000-50-31/logs16/conditional_adversarial_sc16.net','/home/guanmingyang/myproject/LAPGAN/cifar/20000-2000-0-50/log32/conditional_adversarial_sc32.net')
epoch = epoch or 1
for i = 1,300 do  --scale 8
  -- train/test
  adversarial8.train(trainData8) --one epoch
  adversarial8.test(valData8)

  adversarial8.showPNG(valData8)
  --adversarial8.approxParzen(valData8, 200, opt.batchSize)

  sgdState_D.momentum = math.min(sgdState_D.momentum + 0.0008, 0.7)
  sgdState_D.learningRate = math.max(sgdState_D.learningRate / 1.00004, 0.000001)
  sgdState_G.momentum = math.min(sgdState_G.momentum + 0.0008, 0.7)
  sgdState_G.learningRate = math.max(sgdState_G.learningRate / 1.00004, 0.000001)


   if opt.gpu then
      torch.setdefaulttensortype('torch.CudaTensor')
    else
      torch.setdefaulttensortype('torch.FloatTensor')
   end

  epoch = epoch + 1
end


-------------------------------------------------
-- weight upsampling
if opt.ifwbupsampleG1 == 1  then
--[[
  if opt.gpu then
    torch.setdefaulttensortype('torch.CudaTensor')
  else
    torch.setdefaulttensortype('torch.FloatTensor')
  end
]]--
  print('\n8->16 G weights and bias have already upsampled \n')
  local first_filterG16  = torch.FloatTensor(64, 5, 5, 5)
  local second_filterG16 = torch.FloatTensor(64, 64, 5, 5)
  local third_filterG16  = torch.FloatTensor(3, 64, 5, 5)

  local first_filterG8  = torch.FloatTensor(64, 5, 3, 3)
  local second_filterG8 = torch.FloatTensor(64, 64, 3, 3)
  local third_filterG8  = torch.FloatTensor(3, 64, 3, 3)
  -- first conv
  for i=1,64 do
    first_filterG8[i] = pa_G8[3][i]:clone():float()
  end
  for i =1,64 do
    first_filterG16[i] = image.scale(first_filterG8[i], 5, 5) --3*3 -> 7*7
    model_G16:get(8).weight[i] = first_filterG16[i]:clone():cuda()
  end
  for i=1,64 do
    model_G16:get(8).bias[i] = pa_G8[4][i]
  end
  --model_G32:get(8).bias = pa_G16[4]:clone() --bias

  -- second conv
  for i=1,64 do
    second_filterG8[i] = pa_G8[5][i]:clone():float()
  end
  for i =1,64 do
    second_filterG16[i] = image.scale(second_filterG8[i], 5, 5) --3*3 -> 7*7
    model_G16:get(10).weight[i] = second_filterG16[i]:clone():cuda()
  end
  --model_G32:get(10).bias = pa_G16[6]:clone() --bias
  for i=1,64 do
    model_G16:get(10).bias[i] = pa_G8[6][i]
  end

  -- third conv
  for i =1,3 do
    third_filterG8[i] = pa_G8[7][i]:clone():float()
  end
  for i =1,3 do
    third_filterG16[i] = image.scale(third_filterG8[i], 5, 5) --3*3 -> 5*5
    model_G16:get(12).weight[i] = third_filterG16[i]:clone():cuda()
  end
  --model_G32:get(12).bias = pa_G16[8]:clone() --bias
  for i=1,3 do
    model_G16:get(12).bias[i] = pa_G8[6][i]
  end
end

if opt.ifwbupsampleD1 == 1  then

  if opt.gpu then
    torch.setdefaulttensortype('torch.CudaTensor')
  else
    torch.setdefaulttensortype('torch.FloatTensor')
  end

  print('\n8->16 D weights and bias have already upsampled \n')
  local first_filterD16  = torch.FloatTensor(64, 4,  5, 5)
  local second_filterD16 = torch.FloatTensor(64, 64, 5, 5)

  local first_filterD8  = torch.FloatTensor(64, 4,  3, 3)
  local second_filterD8 = torch.FloatTensor(64, 64, 3, 3)
  -- first conv
  for i=1,64 do
    first_filterD8[i] = pa_D8[3][i]:clone():float()
  end
  for i =1,64 do
    first_filterD16[i] = image.scale(first_filterD8[i], 5, 5) --3*3 -> 5*5
    model_D16:get(9).weight[i] = first_filterD16[i]:clone():cuda()
  end
  for i=1,64 do
    model_D16:get(9).bias[i] = pa_D8[4][i]
  end

  -- second conv
  for i=1,64 do
    second_filterD8[i] = pa_D8[5][i]:clone():float()
  end
  for i =1,64 do
    second_filterD16[i] = image.scale(second_filterD8[i], 5, 5) --3*3 -> 5*5
    model_D16:get(11).weight[i] = second_filterD16[i]:clone():cuda()
  end
  --model_G32:get(10).bias = pa_G16[6]:clone() --bias
  for i=1,64 do
    model_D16:get(11).bias[i] = pa_D8[6][i]
  end

end


sgdState_D.momentum = 0.0008
sgdState_D.learningRate = 0.02
sgdState_G.momentum = 0.0008
sgdState_G.learningRate = 0.02
---------------------------------------------------
-- scale 16 train test process
epoch =  1
for i = 1,00 do  --scale 16
  -- train/test
  adversarial16.train(trainData16) --one epoch
  adversarial16.test(valData16)

  adversarial16.showPNG(valData16)
  adversarial16.approxParzen(valData16, 200, opt.batchSize)

  sgdState_D.momentum = math.min(sgdState_D.momentum + 0.0008, 0.7)
  sgdState_D.learningRate = math.max(sgdState_D.learningRate / 1.00004, 0.000001)
  sgdState_G.momentum = math.min(sgdState_G.momentum + 0.0008, 0.7)
  sgdState_G.learningRate = math.max(sgdState_G.learningRate / 1.00004, 0.000001)


   if opt.gpu then
      torch.setdefaulttensortype('torch.CudaTensor')
    else
      torch.setdefaulttensortype('torch.FloatTensor')
   end

  epoch = epoch + 1
end


-------------------------------------------------
-- weight upsampling
if opt.ifwbupsampleG2 == 1  then
--[[
  if opt.gpu then
    torch.setdefaulttensortype('torch.CudaTensor')
  else
    torch.setdefaulttensortype('torch.FloatTensor')
  end
]]--
  print('\n16->32 G weights and bias have already upsampled \n')
  local first_filterG32  = torch.FloatTensor(64, 5, 7, 7)
  local second_filterG32 = torch.FloatTensor(64, 64, 7, 7)
  local third_filterG32  = torch.FloatTensor(3, 64, 7, 7)

  local first_filterG16  = torch.FloatTensor(64, 5, 5, 5)
  local second_filterG16 = torch.FloatTensor(64, 64, 5, 5)
  local third_filterG16  = torch.FloatTensor(3, 64, 5, 5)
  -- first conv
  for i=1,64 do
    first_filterG16[i] = pa_G16[3][i]:clone():float()
  end
  for i =1,64 do
    first_filterG32[i] = image.scale(first_filterG16[i], 7, 7) --3*3 -> 7*7
    model_G32:get(8).weight[i] = first_filterG32[i]:clone():cuda()
  end
  for i=1,64 do
    model_G32:get(8).bias[i] = pa_G16[4][i]
  end
  --model_G32:get(8).bias = pa_G16[4]:clone() --bias

  -- second conv
  for i=1,64 do
    second_filterG16[i] = pa_G16[5][i]:clone():float()
  end
  for i =1,64 do
    second_filterG32[i] = image.scale(second_filterG16[i], 7, 7) --3*3 -> 7*7
    model_G32:get(10).weight[i] = second_filterG32[i]:clone():cuda()
  end
  --model_G32:get(10).bias = pa_G16[6]:clone() --bias
  for i=1,64 do
    model_G32:get(10).bias[i] = pa_G16[6][i]
  end

  -- third conv
  for i =1,3 do
    third_filterG16[i] = pa_G16[7][i]:clone():float()
  end
  for i =1,3 do
    third_filterG32[i] = image.scale(third_filterG16[i], 7, 7) --3*3 -> 5*5
    model_G32:get(12).weight[i] = third_filterG32[i]:clone():cuda()
  end
  --model_G32:get(12).bias = pa_G16[8]:clone() --bias
  for i=1,3 do
    model_G32:get(12).bias[i] = pa_G16[6][i]
  end
end

if opt.ifwbupsampleD2 == 1  then
--[[
  if opt.gpu then
    torch.setdefaulttensortype('torch.CudaTensor')
  else
    torch.setdefaulttensortype('torch.FloatTensor')
  end
]]--
  print('\n16->32 D weights and bias have already upsampled \n')
  local first_filterD32  = torch.FloatTensor(64, 4,  7, 7)
  local second_filterD32 = torch.FloatTensor(64, 64, 7, 7)

  local first_filterD16  = torch.FloatTensor(64, 4,  5, 5)
  local second_filterD16 = torch.FloatTensor(64, 64, 5, 5)
  -- first conv
  for i=1,64 do
    first_filterD16[i] = pa_D16[3][i]:clone():float()
  end
  for i =1,64 do
    first_filterD32[i] = image.scale(first_filterD16[i], 7, 7) --3*3 -> 5*5
    model_D32:get(9).weight[i] = first_filterD32[i]:clone():cuda()
  end
  for i=1,64 do
    model_D32:get(9).bias[i] = pa_D16[4][i]
  end

  -- second conv
  for i=1,64 do
    second_filterD16[i] = pa_D16[5][i]:clone():float()
  end
  for i =1,64 do
    second_filterD32[i] = image.scale(second_filterD16[i], 7, 7) --3*3 -> 5*5
    model_D32:get(11).weight[i] = second_filterD32[i]:clone():cuda()
  end
  --model_G32:get(10).bias = pa_G16[6]:clone() --bias
  for i=1,64 do
    model_D32:get(11).bias[i] = pa_D16[6][i]
  end

end

-- delete the dataset of scale 16
--trainData16 = nil
--valData16 = nil
model_D16 = nil
model_G16 = nil

-- redifine the momentum and lr
sgdState_D.momentum = 0.0008
sgdState_D.learningRate = 0.02
sgdState_G.momentum = 0.0008
sgdState_G.learningRate = 0.02
-----------------------------------------------
-- scale 32 train test process
epoch = 1
for i = 1,00 do  --scale 32

  adversarial32.train(trainData32) --one epoch
  adversarial32.test(valData32)

  adversarial32.showPNG(valData32)
  adversarial32.approxParzen(valData32, 200, opt.batchSize)

  sgdState_D.momentum = math.min(sgdState_D.momentum + 0.0008, 0.7)
  sgdState_D.learningRate = math.max(sgdState_D.learningRate / 1.00004, 0.000001)
  sgdState_G.momentum = math.min(sgdState_G.momentum + 0.0008, 0.7)
  sgdState_G.learningRate = math.max(sgdState_G.learningRate / 1.00004, 0.000001)

   if opt.gpu then
      torch.setdefaulttensortype('torch.CudaTensor')
    else
      torch.setdefaulttensortype('torch.FloatTensor')
    end
  epoch = epoch +1
end

--------------------------------------------------
-- plot the logger
if opt.mode == 8 or opt.mode == 1 then
  trainLogger8:style{['% mean class accuracy (train set)'] = '+-'}
  testLogger8:style{['% mean class accuracy (test set)'] = '+-' }
  avdistanceLogger8:style{['average distance'] = '+-' }
  maxdistanceLogger8:style{['max distance'] = '+-' }
  mindistanceLogger8:style{['min distance'] = '+-'}

  trainLogger8:plot()
  testLogger8:plot()
  avdistanceLogger8:plot()
  mindistanceLogger8:plot()
  maxdistanceLogger8:plot()
end

if opt.mode == 16 or opt.mode == 1  then
  trainLogger16:style{['% mean class accuracy (train set)'] = '+-'}
  testLogger16:style{['% mean class accuracy (test set)'] = '+-' }
  avdistanceLogger16:style{['average distance'] = '+-' }
  maxdistanceLogger16:style{['max distance'] = '+-' }
  mindistanceLogger16:style{['min distance'] = '+-' }

  trainLogger16:plot()
  testLogger16:plot()
  avdistanceLogger16:plot()
  mindistanceLogger16:plot()
  maxdistanceLogger16:plot()

end
if opt.mode == 32 or opt.mode == 1  then
  trainLogger32:style{['% mean class accuracy (train set)'] = '+-' }
  testLogger32:style{['% mean class accuracy (test set)'] = '+-' }
  avdistanceLogger32:style{['average distance'] = '+-' }
  maxdistanceLogger32:style{['max distance'] = '+-' }
  mindistanceLogger32:style{['min distance'] = '+-' }
  avdistanceLogger32:plot()
  mindistanceLogger32:plot()
  maxdistanceLogger32:plot()
  trainLogger32:plot()
  testLogger32:plot()
end