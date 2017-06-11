require 'torch'
require 'optim'
require 'pl'
require 'paths'
require 'image'

local adversarial_sc16 = {}

geometry16 = {3,16,16 }
noiseDim16 = {1,16,16}
condDim1_sc16 = 10
condDim2_sc16 = {3,16,16}
-- training function
function adversarial_sc16.train(dataset, N)

  --epoch = epoch or 1

  local N = N or dataset:size() --if N is not nil,then return N ;otherwise return dataset:size()
  local time = sys.clock()
  local dataBatchSize = opt.batchSize / 2 --64

  local inputs = torch.Tensor(opt.batchSize, geometry16[1], geometry16[2], geometry16[3]) --128*3*16*16
  local targets = torch.Tensor(opt.batchSize) --128维
  --noise inputs
  local noise_inputs
  noise_inputs = torch.Tensor(opt.batchSize,noiseDim16[1],noiseDim16[2],noiseDim16[3]) --128*1*16*16
  -- conditon inputs
  local cond_inputs1
  local cond_inputs2
  cond_inputs1 = torch.Tensor(opt.batchSize, condDim1_sc16)  --default：conDim1 = 10
  cond_inputs2 = torch.Tensor(opt.batchSize, condDim2_sc16[1], condDim2_sc16[2], condDim2_sc16[3]) --condDim2 = {3,16,16}

  -- do one epoch
  print('\n<trainer><scale-16> on training set:')
  print("<trainer><scale-16> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ' lr = ' .. sgdState_D.learningRate .. ', momentum = ' .. sgdState_D.momentum .. ']')

  for t = 1,N,dataBatchSize*opt.K do
    -- dataBatchSize = opt.batchSize / 2 default:128/2 = 64
    -- opt.K: number of iterations to optimize D for. default=1
    -- so step = 64

    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of discriminator
    local fevalD = function(x)
      collectgarbage()
      if x ~= parameters_D16 then -- get new parameters
        parameters_D16:copy(x)
      end

      gradParameters_D16:zero() -- reset gradients

      --  forward pass
      local outputs = model_D16:forward({inputs, cond_inputs1, cond_inputs2})
      local f = criterion16:forward(outputs, targets)  -- f is the loss function

      -- backward pass 
      local df_do = criterion16:backward(outputs, targets)
      model_D16:backward({inputs, cond_inputs1, cond_inputs2}, df_do)

      -- penalties (L1 and L2):
      if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
        local norm,sign= torch.norm,torch.sign
        -- Loss:
        f = f + opt.coefL1 * norm(parameters_D16,1)
        f = f + opt.coefL2 * norm(parameters_D16,2)^2/2
        -- Gradients:
        gradParameters_D16:add( sign(parameters_D16):mul(opt.coefL1) + parameters_D16:clone():mul(opt.coefL2) )
      end
      -- update confusion (add 1 since classes are binary)
      for i = 1,opt.batchSize do
        local c
        if outputs[i][1] > 0.5 then c = 2 else c = 1 end
        confusion16:add(c, targets[i]+1) --前一半batchbize 64为1 后一半为0
      end

      return f,gradParameters_D16  --  E and dE/dW 返回loss 还有loss对参数的梯度
    end

    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of generator 
    local fevalG = function(x)
      collectgarbage()
      if x ~= parameters_G16 then -- get new parameters
        parameters_G16:copy(x)
      end
      
      gradParameters_G16:zero() -- reset gradients

      -- G -> D
      -- forward pass
      local samples = model_G16:forward({noise_inputs, cond_inputs1, cond_inputs2})
      local outputs = model_D16:forward({samples, cond_inputs1, cond_inputs2})
      local f = criterion16:forward(outputs, targets)

      --  backward pass
      local df_samples = criterion16:backward(outputs, targets)
      model_D16:backward({samples, cond_inputs1, cond_inputs2}, df_samples)
      local df_do = model_D16.gradInput[1]
      model_G16:backward({noise_inputs, cond_inputs1, cond_inputs2}, df_do)

      -- penalties (L1 and L2):
      if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
        local norm,sign= torch.norm,torch.sign
        -- Loss:
        f = f + opt.coefL1 * norm(parameters_D16,1)
        f = f + opt.coefL2 * norm(parameters_D16,2)^2/2
        -- Gradients:
        gradParameters_G16:add( sign(parameters_G16):mul(opt.coefL1) + parameters_G16:clone():mul(opt.coefL2) )
      end

      return f,gradParameters_G16
    end

    ----------------------------------------------------------------------
    -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    -- Get half a minibatch of real, half fake
    for j=1,opt.K do
      -- (1.1) Real data
      -- inputs前64个为真differ
      local k = 1
      for i = t,math.min(t+dataBatchSize-1,dataset:size()) do
        -- load new sample
        local idx = math.random(dataset:size()) -- create a int on [1,N]
        local sample = dataset[idx]
        inputs[k] = sample[1]:clone()           -- load the real diff
        cond_inputs1[k] = sample[2]:clone()     -- load the label vector
        cond_inputs2[k] = sample[3]:clone()     -- load the coarse image(cond2)
        k = k + 1
      end
      targets[{{1,dataBatchSize}}]:fill(1)  -- 1-64 fills 1

      -- (1.2) Sampled data
      -- inputs[64-128]是生成的differ
      noise_inputs:uniform(-1, 1)
      for i = dataBatchSize+1,opt.batchSize do
        local idx = math.random(dataset:size())
        local sample = dataset[idx]
        cond_inputs1[i] = sample[2]:clone()     -- load the label vector
        cond_inputs2[i] = sample[3]:clone()     -- load the coarse image(cond2)
      end
      local samples = model_G16:forward({noise_inputs[{{dataBatchSize+1,opt.batchSize}}], cond_inputs1[{{dataBatchSize+1,opt.batchSize}}], cond_inputs2[{{dataBatchSize+1,opt.batchSize}}]})
      for i = 1, dataBatchSize do
        inputs[k] = samples[i]:clone()          -- load the diff_gen
        k = k + 1
      end
      targets[{{dataBatchSize+1,opt.batchSize}}]:fill(0) --65-128 fills 0

      optim.sgd(fevalD, parameters_D16, sgdState_D) -- parameters_D16 is the input of fevalD
      -- sgd(opfunc,x,[ config],[ state])
      -- opfunc: a function that takes a single input X, the point of a evaluation, and returns f(X) and df/dX
      -- X     : the initial input of above opfunc
      -- config: sgdState_D = { learningRate = opt.learningRate, momentum = opt.momentum }
    end -- end for K

    ----------------------------------------------------------------------
    -- (2) Update G network: maximize log(D(G(z)))
    noise_inputs:uniform(-1, 1)
    for i = 1,opt.batchSize do
      local idx = math.random(dataset:size())
      local sample = dataset[idx]
      cond_inputs1[i] = sample[2]:clone() -- load the label vector
      cond_inputs2[i] = sample[3]:clone() -- load the coarse image(cond2)
    end
    targets:fill(1)
    optim.sgd(fevalG, parameters_G16, sgdState_G)

    -- disp progress 显示进度条
    xlua.progress(t, N)
  end -- end for loop over dataset

  -- time taken
  time = sys.clock() - time
  time = time / dataset:size()
  print("\n<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  print(confusion16)
  --logger16:add{['% mean class accuracy (train set)'] = confusion16.totalValid * 100}
  trainLogger16:add{['% mean class accuracy (train set)'] = confusion16.totalValid * 100}
  confusion16:zero()

  -- save/log current net
  if epoch % opt.saveFreq == 0 then
    local filename = paths.concat(opt.save16,'epoch',epoch,  'conditional_adversarial_sc16.net') --当前目录/opt.save16/conditional_adversarial_sc16.net
    os.execute('mkdir -p ' .. sys.dirname(filename)) --创建文件所在的文件夹
    --[[
    if paths.filep(filename) then  -- if the file already exits
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old') --rename the old file
    end
    ]]--
    print('<trainer> saving network to '..filename)
    torch.save(filename, {D = model_D16, G = model_G16, E = model_E16, opt = opt})
  end

end

-- test function
function adversarial_sc16.test(dataset, N)

  local time = sys.clock()
  local N = N or dataset:size()

  local inputs = torch.Tensor(opt.batchSize, geometry16[1], geometry16[2], geometry16[3]) --128*3*16*16

  -- noise input
  local noise_inputs
  noise_inputs = torch.Tensor(opt.batchSize, noiseDim16[1],noiseDim16[2],noiseDim16[3]) --128*1*16*16
  -- conditon inputs
  local cond_inputs1
  local cond_inputs2
  cond_inputs1 = torch.Tensor(opt.batchSize, condDim1_sc16)  --default：conDim1 = 10
  cond_inputs2 = torch.Tensor(opt.batchSize, condDim2_sc16[1], condDim2_sc16[2], condDim2_sc16[3]) --condDim2 = {3,16,16}

  print('\n<trainer><scale-16> on testing set:')

  for t = 1,N,opt.batchSize do --128为一组
    -- display progress
    xlua.progress(t, N)
    ----------------------------------------------------------------------
    -- (1) Real data
    local targets = torch.ones(opt.batchSize) --128维的1

    local k = 1
    for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
      local idx = math.random(dataset:size())
      local sample = dataset[idx]
      inputs[k] = sample[1]:clone() --load the diff
      cond_inputs1[k] = sample[2]:clone() -- load the label
      cond_inputs2[k] = sample[3]:clone() -- load the corase
      k = k + 1
    end
    local preds = model_D16:forward({inputs, cond_inputs1, cond_inputs2}) -- get predictions from D
    -- add to confusion matrix
    for i = 1,opt.batchSize do
      local c
      if preds[i][1] > 0.5 then c = 2 else c = 1 end
      confusion16:add(c, targets[i] + 1) --期望结果是（2,2）位置加一
      -- ConfusionMatrix:add(prediction, target)
      --真实标签为target，预测标签为prediction 在矩阵相应位置加1 即统计判断情况
    end

    ----------------------------------------------------------------------
    -- (2) Generated data (don't need this really, since no 'validation' generations)
    noise_inputs:uniform(-1, 1)
    local targets = torch.zeros(opt.batchSize) --128维的0
    for i = 1,opt.batchSize do
      sample = dataset[math.random(dataset:size())]
      cond_inputs1[i] = sample[2]:clone()  --load the label
      cond_inputs2[i] = sample[3]:clone()  --load the corase
    end
    local samples = model_G16:forward({noise_inputs, cond_inputs1, cond_inputs2})
    local preds = model_D16:forward({samples, cond_inputs1, cond_inputs2}) -- get predictions from D
    -- add to confusion matrix
    for i = 1,opt.batchSize do
      local c
      if preds[i][1] > 0.5 then c = 2 else c = 1 end
      confusion16:add(c, targets[i] + 1) --D的期望结果是(1,1)加1 G的期望结果是（1,2）加1
    end
  end -- end loop over dataset

  -- timing
  time = sys.clock() - time
  time = time / dataset:size()
  print("\n<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')
  -- print confusion matrix
  print(confusion16)
  --logger16:add{['% mean class accuracy (test set)'] = confusion16.totalValid * 100}
  testLogger16:add{['% mean class accuracy (test set)'] = confusion16.totalValid * 100}
  confusion16:zero()

  if opt.gpu then
    torch.setdefaulttensortype('torch.CudaTensor')
  else
    torch.setdefaulttensortype('torch.FloatTensor')
  end
  return cond_inputs
end

-- plot the real/gen differ real/gen fine
function adversarial_sc16.showPNG(dataset,n) -- plot the png

  n = n or 60
  -- noise input
  local noise_inputs
  noise_inputs = torch.Tensor(opt.batchSize, noiseDim16[1],noiseDim16[2],noiseDim16[3]) --128*100
  -- conditon inputs
  local cond_inputs1
  local cond_inputs2
  cond_inputs1 = torch.Tensor(opt.batchSize, condDim1_sc16)  --default：conDim1 = 10
  cond_inputs2 = torch.Tensor(opt.batchSize, condDim2_sc16[1], condDim2_sc16[2], condDim2_sc16[3]) --condDim2 = {3,16,16}
  -- real/gen fine
  local realfines = torch.Tensor(opt.batchSize, condDim2_sc16[1], condDim2_sc16[2], condDim2_sc16[3])
  local genfines = torch.Tensor(opt.batchSize, condDim2_sc16[1], condDim2_sc16[2], condDim2_sc16[3])
  -- real differ
  local realdiffers = torch.Tensor(opt.batchSize, geometry16[1], geometry16[2], geometry16[3]) --128*3*16*16

  for i = 1,n  do
    local idx = math.random(dataset:size())
    local sample = dataset[idx]
    realdiffers[i] = sample[1]:clone() --load the diff
    cond_inputs1[i] = sample[2]:clone() -- load the label
    cond_inputs2[i] = sample[3]:clone() -- load the corase
    realfines[i] = sample[4]:clone() -- load the real fine
  end
  noise_inputs:uniform(-1,1)
  local gendiffers = model_G16:forward({noise_inputs, cond_inputs1, cond_inputs2})
  for i=1,n do
    genfines[i] = torch.add(cond_inputs2[i],1,gendiffers[i])
  end
--[[
  print('\n')
  print(torch.type(cond_inputs1))
  print('\n')
  print(torch.getdefaulttensortype())
  ]]--
  local to_plot = {}
  -- plot real differ
  for i=1,n do
    to_plot[i] = realdiffers[i]:float()
  end
  local fname = paths.concat(opt.save16, 'real_differs-epoch-' .. epoch .. '.png')
  torch.setdefaulttensortype('torch.FloatTensor')
  image.save(fname, image.toDisplayTensor{input=to_plot, scaleeach=true})
  -- plot gen differ
  for i=1,n do
    to_plot[i] = gendiffers[i]:float()
  end
  local fname = paths.concat(opt.save16, 'gen_differs-epoch-' .. epoch .. '.png')
  torch.setdefaulttensortype('torch.FloatTensor')
  image.save(fname, image.toDisplayTensor{input=to_plot, scaleeach=true})
  --plot gen fines
  for i=1,n do
    to_plot[i] = genfines[i]:float()
  end
  local fname = paths.concat(opt.save16, 'gen_fines-epoch-' .. epoch .. '.png')
  torch.setdefaulttensortype('torch.FloatTensor')
  image.save(fname, image.toDisplayTensor{input=to_plot, scaleeach=true})
  --plot real fines
  for i=1,n do
    to_plot[i] = realfines[i]:float()
  end
  local fname = paths.concat(opt.save16, 'real_fines-epoch-' .. epoch .. '.png')
  torch.setdefaulttensortype('torch.FloatTensor')
  image.save(fname, image.toDisplayTensor{input=to_plot, scaleeach=true})

end

-- Unnormalized parzen window type estimate (used to track performance during training)
-- Really just a nearest neighbours of ground truth to multiple generations
function adversarial_sc16.approxParzen(dataset, nsamples, nneighbors) --valdata,200,128

  best_dist = best_dist or 1e10
  print('\n<trainer><scale-16> evaluating approximate parzen ')

  if opt.gpu then
      torch.setdefaulttensortype('torch.CudaTensor')
    else
      torch.setdefaulttensortype('torch.FloatTensor')
   end

  local inputs = torch.Tensor(opt.batchSize, geometry16[1], geometry16[2], geometry16[3]) --128*3*16*16
  -- noise input
  local noise_inputs
  noise_inputs = torch.Tensor(opt.batchSize, noiseDim16[1],noiseDim16[2],noiseDim16[3]) --128*1*16*16
  -- conditon inputs
  local cond_inputs1
  local cond_inputs2
  cond_inputs1 = torch.Tensor(opt.batchSize, condDim1_sc16)  --default：condDim1_sc16 = 10
  cond_inputs2 = torch.Tensor(opt.batchSize, condDim2_sc16[1], condDim2_sc16[2], condDim2_sc16[3]) --condDim2_sc16 = {3,16,16}

  local distances = torch.Tensor(nsamples) --200


  for n = 1,nsamples do --200次
    --show the progress
    xlua.progress(n, nsamples)
    --sampling
    local sample = dataset[math.random(dataset:size())]
    local fine = sample[4]:type(torch.getdefaulttensortype()) --load the fine
    -- fill the input:noise cond1 cond2
    noise_inputs:uniform(-1, 1)
    for i = 1,nneighbors do    --128个相同的粗图像 128*3*16*16
      cond_inputs1[i] = sample[2]:clone()  --the label
      cond_inputs2[i] = sample[3]:clone()  --the corase
    end
    -- generate the fake image
      local neighbors = model_G16:forward({noise_inputs, cond_inputs1,cond_inputs2}) --输入相同的粗图像和不同的噪声
    neighbors:add(cond_inputs2) --加上粗图像 得到细图像
    -- compute distance
    local dist = 1e10
    for i = 1,nneighbors do
      dist = math.min(torch.dist(neighbors[i], fine), dist) --128张图片和真图片的最小欧式距离
    end
    distances[n] = dist
  end

  print('<scale16> average distance = ' .. distances:mean())
  print('<scale16>   max   distance = ' .. distances:max())
  print('<scale16>   min   distance = ' .. distances:min())
  --print(type(distances:max()))
  --local avedist = distances:mean():float()
  avdistanceLogger16:add{['average distance'] = distances:mean() }
  maxdistanceLogger16:add{['max distance'] = distances:max()}
  mindistanceLogger16:add{['min distance'] = distances:min()}
  -- save/log current net
  if distances:mean() < best_dist then
    best_dist = distances:mean()
    local filename = paths.concat(opt.save16, 'conditional_adversarial_sc16.bestnet')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    --[[
    if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end
    ]]--
    print('<trainer><scale-16> saving network to '..filename)
    torch.save(filename, {D = model_D16, G = model_G16, E = model_E16, opt = opt})
  end

  return distances
end

return adversarial_sc16
