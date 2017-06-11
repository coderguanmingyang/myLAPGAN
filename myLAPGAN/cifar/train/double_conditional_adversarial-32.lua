require 'torch'
require 'optim'
require 'pl'
require 'paths'
require 'image'

local adversarial_sc32 = {}

geometry32 = {3,32,32 }
noiseDim32 = {1,32,32}
condDim1_sc32 = 10
condDim2_sc32 = {3,32,32}
-- training function
function adversarial_sc32.train(dataset, N)

  -- epoch = epoch or 1

  local N = N or dataset:size() --if N is not nil,then return N ;otherwise return dataset:size()
  local time = sys.clock()
  local dataBatchSize = opt.batchSize / 2 --64

  local inputs = torch.Tensor(opt.batchSize, geometry32[1], geometry32[2], geometry32[3]) --128*3*32*32
  local targets = torch.Tensor(opt.batchSize) --128维
  --noise inputs
  local noise_inputs
  noise_inputs = torch.Tensor(opt.batchSize, noiseDim32[1],noiseDim32[2],noiseDim32[3]) --128*1*32*32
  -- conditon inputs
  local cond_inputs1
  local cond_inputs2
  cond_inputs1 = torch.Tensor(opt.batchSize, condDim1_sc32)  --default：conDim1 = 10
  cond_inputs2 = torch.Tensor(opt.batchSize, condDim2_sc32[1], condDim2_sc32[2], condDim2_sc32[3]) --condDim2 = {3,32,32}

  -- do one epoch
  print('\n<trainer><scale-32> on training set:')
  print("<trainer><scale-32> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ' lr = ' .. sgdState_D.learningRate .. ', momentum = ' .. sgdState_D.momentum .. ']')

  for t = 1,N,dataBatchSize*opt.K do
    -- dataBatchSize = opt.batchSize / 2 default:128/2 = 64
    -- opt.K: number of iterations to optimize D for. default=1
    -- so step = 64

    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of discriminator
    local fevalD = function(x)

      collectgarbage()
      if x ~= parameters_D32 then -- get new parameters
        parameters_D32:copy(x)
      end

      gradParameters_D32:zero() -- reset gradients

      --  forward pass
      local outputs = model_D32:forward({inputs, cond_inputs1, cond_inputs2})
      local f = criterion32:forward(outputs, targets)  -- f is the loss function

      -- backward pass 
      local df_do = criterion32:backward(outputs, targets)
      model_D32:backward({inputs, cond_inputs1, cond_inputs2}, df_do)

      -- penalties (L1 and L2):
      if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
        local norm,sign= torch.norm,torch.sign
        -- Loss:
        f = f + opt.coefL1 * norm(parameters_D32,1)
        f = f + opt.coefL2 * norm(parameters_D32,2)^2/2
        -- Gradients:
        gradParameters_D32:add( sign(parameters_D32):mul(opt.coefL1) + parameters_D32:clone():mul(opt.coefL2) )
      end
      -- update confusion (add 1 since classes are binary)
      for i = 1,opt.batchSize do
        local c
        if outputs[i][1] > 0.5 then c = 2 else c = 1 end
        confusion32:add(c, targets[i]+1) --前一半batchbize 64为1 后一半为0
      end

      return f,gradParameters_D32  --  E and dE/dW 返回loss 还有loss对参数的梯度
    end

    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of generator 
    local fevalG = function(x)
      collectgarbage()
      if x ~= parameters_G32 then -- get new parameters
        parameters_G32:copy(x)
      end
      
      gradParameters_G32:zero() -- reset gradients

      -- G -> D
      -- forward pass
      local samples = model_G32:forward({noise_inputs, cond_inputs1, cond_inputs2})
      local outputs = model_D32:forward({samples, cond_inputs1, cond_inputs2})
      local f = criterion32:forward(outputs, targets)

      --  backward pass
      local df_samples = criterion32:backward(outputs, targets)
      model_D32:backward({samples, cond_inputs1, cond_inputs2}, df_samples)
      local df_do = model_D32.gradInput[1]
      model_G32:backward({noise_inputs, cond_inputs1, cond_inputs2}, df_do)

      -- penalties (L1 and L2):
      if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
        local norm,sign= torch.norm,torch.sign
        -- Loss:
        f = f + opt.coefL1 * norm(parameters_D32,1)
        f = f + opt.coefL2 * norm(parameters_D32,2)^2/2
        -- Gradients:
        gradParameters_G32:add( sign(parameters_G32):mul(opt.coefL1) + parameters_G32:clone():mul(opt.coefL2) )
      end

      return f,gradParameters_G32
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
      --noise_inputs:normal(0, 0.1)
      for i = dataBatchSize+1,opt.batchSize do
        local idx = math.random(dataset:size())
        local sample = dataset[idx]
        cond_inputs1[i] = sample[2]:clone()     -- load the label vector
        cond_inputs2[i] = sample[3]:clone()     -- load the coarse image(cond2)
      end
      local samples = model_G32:forward({noise_inputs[{{dataBatchSize+1,opt.batchSize}}], cond_inputs1[{{dataBatchSize+1,opt.batchSize}}], cond_inputs2[{{dataBatchSize+1,opt.batchSize}}]})
      for i = 1, dataBatchSize do
        inputs[k] = samples[i]:clone()          -- load the diff_gen
        k = k + 1
      end
      targets[{{dataBatchSize+1,opt.batchSize}}]:fill(0) --65-128 fills 0

      optim.sgd(fevalD, parameters_D32, sgdState_D) -- parameters_D32 is the input of fevalD
      -- sgd(opfunc,x,[ config],[ state])
      -- opfunc: a function that takes a single input X, the point of a evaluation, and returns f(X) and df/dX
      -- X     : the initial input of above opfunc
      -- config: sgdState_D = { learningRate = opt.learningRate, momentum = opt.momentum }
    end -- end for K

    ----------------------------------------------------------------------
    -- (2) Update G network: maximize log(D(G(z)))
    noise_inputs:uniform(-1, 1)
    --noise_inputs:normal(0, 0.1)
    for i = 1,opt.batchSize do
      local idx = math.random(dataset:size())
      local sample = dataset[idx]
      cond_inputs1[i] = sample[2]:clone() -- load the label vector
      cond_inputs2[i] = sample[3]:clone() -- load the coarse image(cond2)
    end
    targets:fill(1)
    optim.sgd(fevalG, parameters_G32, sgdState_G)

    -- disp progress 显示进度条
    xlua.progress(t, N)
  end -- end for loop over dataset

  -- time taken
  time = sys.clock() - time
  time = time / dataset:size()
  print("\n<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  print(confusion32)
  --logger32:add{['% mean class accuracy (train set)'] = confusion32.totalValid * 100}
  trainLogger32:add{['% mean class accuracy (train set)'] = confusion32.totalValid * 100}
  confusion32:zero()

  -- save/log current net
  if epoch % opt.saveFreq == 0 then
    local filename = paths.concat(opt.save32,'epoch',epoch,'conditional_adversarial_sc32.net') --当前目录/opt.save32/conditional_adversarial_sc32.net

    os.execute('mkdir -p ' .. sys.dirname(filename)) --创建文件所在的文件夹
    --[[
    if paths.filep(filename) then  -- if the file already exits
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old') --rename the old file
    end
    ]]--
    print('<trainer> saving network to '..filename)
    torch.save(filename, {D = model_D32, G = model_G32, E = model_E32, opt = opt})
  end

  -- next epoch
  --epoch = epoch + 1
end

-- test function
function adversarial_sc32.test(dataset, N)

  local time = sys.clock()
  local N = N or dataset:size()

  local inputs = torch.Tensor(opt.batchSize, geometry32[1], geometry32[2], geometry32[3]) --128*3*32*32

  -- noise input
  local noise_inputs
  noise_inputs = torch.Tensor(opt.batchSize, noiseDim32[1],noiseDim32[2],noiseDim32[3]) --128*1*32*32
  -- conditon inputs
  local cond_inputs1
  local cond_inputs2
  cond_inputs1 = torch.Tensor(opt.batchSize, condDim1_sc32)  --default：conDim1 = 10
  cond_inputs2 = torch.Tensor(opt.batchSize, condDim2_sc32[1], condDim2_sc32[2], condDim2_sc32[3]) --condDim2 = {3,32,32}

  print('\n<trainer><scale-32> on testing set:')

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
    local preds = model_D32:forward({inputs, cond_inputs1, cond_inputs2}) -- get predictions from D
    -- add to confusion matrix
    for i = 1,opt.batchSize do
      local c
      if preds[i][1] > 0.5 then c = 2 else c = 1 end
      confusion32:add(c, targets[i] + 1) --期望结果是（2,2）位置加一
      -- ConfusionMatrix:add(prediction, target)
      --真实标签为target，预测标签为prediction 在矩阵相应位置加1 即统计判断情况
    end

    ----------------------------------------------------------------------
    -- (2) Generated data (don't need this really, since no 'validation' generations)
    noise_inputs:uniform(-1, 1)
    --noise_inputs:normal(0, 0.1)
    local targets = torch.zeros(opt.batchSize) --128维的0
    for i = 1,opt.batchSize do
      sample = dataset[math.random(dataset:size())]
      cond_inputs1[i] = sample[2]:clone()  --load the label
      cond_inputs2[i] = sample[3]:clone()  --load the corase
    end
    local samples = model_G32:forward({noise_inputs, cond_inputs1, cond_inputs2})
    local preds = model_D32:forward({samples, cond_inputs1, cond_inputs2}) -- get predictions from D
    -- add to confusion matrix
    for i = 1,opt.batchSize do
      local c
      if preds[i][1] > 0.5 then c = 2 else c = 1 end
      confusion32:add(c, targets[i] + 1) --D的期望结果是(1,1)加1 G的期望结果是（2,1）加1
    end
  end -- end loop over dataset

  -- timing
  time = sys.clock() - time
  time = time / dataset:size()
  print("\n<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  print(confusion32)
  --logger32:add{['% mean class accuracy (test set)'] = confusion32.totalValid * 100}
  testLogger32:add{['% mean class accuracy (test set)'] = confusion32.totalValid * 100}
  confusion32:zero()

  if opt.gpu then
    torch.setdefaulttensortype('torch.CudaTensor')
  else
    torch.setdefaulttensortype('torch.FloatTensor')
  end
  return cond_inputs
end

-- plot the real/gen differ real/gen fine
function adversarial_sc32.showPNG(dataset,n) -- plot the png

  n = n or 60
  -- noise input
  local noise_inputs
  noise_inputs = torch.Tensor(opt.batchSize, noiseDim32[1],noiseDim32[2],noiseDim32[3]) --128*100
  -- conditon inputs
  local cond_inputs1
  local cond_inputs2
  cond_inputs1 = torch.Tensor(opt.batchSize, condDim1_sc32)  --default：conDim1 = 10
  cond_inputs2 = torch.Tensor(opt.batchSize, condDim2_sc32[1], condDim2_sc32[2], condDim2_sc32[3]) --condDim2 = {3,32,32}
  -- real/gen fine
  local realfines = torch.Tensor(opt.batchSize, condDim2_sc32[1], condDim2_sc32[2], condDim2_sc32[3])
  local genfines = torch.Tensor(opt.batchSize, condDim2_sc32[1], condDim2_sc32[2], condDim2_sc32[3])
  -- real differ
  local realdiffers = torch.Tensor(opt.batchSize, geometry32[1], geometry32[2], geometry32[3]) --128*3*32*32

  for i = 1,n  do
    local idx = math.random(dataset:size())
    local sample = dataset[idx]
    realdiffers[i] = sample[1]:clone() --load the diff
    cond_inputs1[i] = sample[2]:clone() -- load the label
    cond_inputs2[i] = sample[3]:clone() -- load the corase
    realfines[i] = sample[4]:clone() -- load the real fine
  end
  noise_inputs:uniform(-1,1)
  --noise_inputs:normal(0,0.1)
  local gendiffers = model_G32:forward({noise_inputs, cond_inputs1, cond_inputs2})
  for i=1,n do
    genfines[i] = torch.add(cond_inputs2[i],1,gendiffers[i])
  end

  local to_plot = {}
  -- plot real differ
  for i=1,n do
    to_plot[i] = realdiffers[i]:float()
  end
  local fname = paths.concat(opt.save32, 'real_differs-epoch-' .. epoch .. '.png')
  torch.setdefaulttensortype('torch.FloatTensor')
  image.save(fname, image.toDisplayTensor{input=to_plot, scaleeach=true})
  -- plot gen differ
  for i=1,n do
    to_plot[i] = gendiffers[i]:float()
  end
  local fname = paths.concat(opt.save32, 'gen_differs-epoch-' .. epoch .. '.png')
  torch.setdefaulttensortype('torch.FloatTensor')
  image.save(fname, image.toDisplayTensor{input=to_plot, scaleeach=true})
  --plot gen fines
  for i=1,n do
    to_plot[i] = genfines[i]:float()
  end
  local fname = paths.concat(opt.save32, 'gen_fines-epoch-' .. epoch .. '.png')
  torch.setdefaulttensortype('torch.FloatTensor')
  image.save(fname, image.toDisplayTensor{input=to_plot, scaleeach=true})
  --plot real fines
  for i=1,n do
    to_plot[i] = realfines[i]:float()
  end
  local fname = paths.concat(opt.save32, 'real_fines-epoch-' .. epoch .. '.png')
  torch.setdefaulttensortype('torch.FloatTensor')
  image.save(fname, image.toDisplayTensor{input=to_plot, scaleeach=true})

end

-- Unnormalized parzen window type estimate (used to track performance during training)
-- Really just a nearest neighbours of ground truth to multiple generations
function adversarial_sc32.approxParzen(dataset, nsamples, nneighbors) --valdata,200,128

  best_dist = best_dist or 1e10
  print('\n<trainer><scale-32> evaluating approximate parzen ')

  if opt.gpu then
      torch.setdefaulttensortype('torch.CudaTensor')
    else
      torch.setdefaulttensortype('torch.FloatTensor')
  end

  local noise_inputs
  noise_inputs = torch.Tensor(opt.batchSize, noiseDim32[1],noiseDim32[2],noiseDim32[3]) --128*100
  -- conditon inputs
  local cond_inputs1
  local cond_inputs2
  cond_inputs1 = torch.Tensor(opt.batchSize, condDim1_sc32)  --default：conDim1 = 10
  cond_inputs2 = torch.Tensor(opt.batchSize, condDim2_sc32[1], condDim2_sc32[2], condDim2_sc32[3]) --condDim2 = {3,32,32}
    -- real/gen fine
  local realfine = torch.Tensor(condDim2_sc32[1], condDim2_sc32[2], condDim2_sc32[3])
  local genfines = torch.Tensor(opt.batchSize, condDim2_sc32[1], condDim2_sc32[2], condDim2_sc32[3])
  -- real differ
  local realdiffers = torch.Tensor(opt.batchSize, geometry32[1], geometry32[2], geometry32[3]) --128*3*32*32

  local distances = torch.Tensor(nsamples) --200

  for n = 1,nsamples do --200次
    --show the progress
   xlua.progress(n, nsamples)
    -- noise input
   local  idx = math.random(dataset:size())

    for i = 1,128  do
      local sample = dataset[idx]
      realdiffers[i] = sample[1]:clone() --load the diff
      cond_inputs1[i] = sample[2]:clone() -- load the label
      cond_inputs2[i] = sample[3]:clone() -- load the corase
      realfine = sample[4]:clone():type(torch.getdefaulttensortype()) -- load the real fine
    end
    noise_inputs:uniform(-1,1)
    --noise_inputs:normal(-1,1)
    local gendiffers = model_G32:forward({noise_inputs, cond_inputs1, cond_inputs2})
    for i=1,128 do
      genfines[i] = torch.add(cond_inputs2[i],1,gendiffers[i])
    end
    -- compute distance
    local dist = 1e10
    for i = 1,nneighbors do
      dist = math.min(torch.dist(genfines[i], realfine), dist) --128张图片和真图片的最小欧式距离
    end
    distances[n] = dist
  end

  print('<scale32> average distance = ' .. distances:mean())
  print('<scale32>   max   distance = ' .. distances:max())
  print('<scale32>   min   distance = ' .. distances:min())
  avdistanceLogger32:add{['average distance'] = distances:mean() }
  maxdistanceLogger32:add{['max distance'] = distances:max() }
  mindistanceLogger32:add{['min distance'] = distances:min()}
  -- save/log current net
  if distances:mean() < best_dist then
    best_dist = distances:mean()
    local filename = paths.concat(opt.save32, 'conditional_adversarial_sc32.bestnet')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    --[[
    if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end
    ]]--
    print('<trainer><scale-32> saving network to '..filename)
    torch.save(filename, {D = model_D32, G = model_G32, E = model_E32, opt = opt})
  end

  return distances
end

return adversarial_sc32
