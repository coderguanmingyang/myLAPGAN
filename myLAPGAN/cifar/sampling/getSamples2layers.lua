--
-- Created by IntelliJ IDEA.
-- User: PC
-- Date: 2017/5/5
-- Time: 15:49
-- To change this template use File | Settings | File Templates.
--
require 'torch'
require 'optim'
require 'pl'
require 'paths'
require 'image'

local getSamples16_32 = {}

condDim2_sc16 = {3, 16, 16 }

function getSamples16_32.genImage(dataset, G16, G32)

  torch.setdefaulttensortype('torch.CudaTensor')

  local tmp16 = torch.load(G16)
  print('G16 load succsefully')
  local tmp32 = torch.load(G32)
  print('G32 load succsefully')
  model_G16 = tmp16.G
  model_G32 = tmp32.G
  -- noise input
  local noise_inputs16
  noise_inputs16 = torch.Tensor(batchsize, noiseDim16[1], noiseDim16[2], noiseDim16[3]) --128*1*16*16
  local noise_inputs32
  noise_inputs32 = torch.Tensor(batchsize, noiseDim32[1], noiseDim32[2], noiseDim32[3]) --128*1*32*32
  -- conditon inputs
  local cond_inputs1
  local cond_inputs2
  cond_inputs1 = torch.Tensor(batchsize, 10)  --default：conDim1 = 10
  cond_inputs2 = torch.Tensor(batchsize, condDim2_sc16[1], condDim2_sc16[2], condDim2_sc16[3]) --condDim2 = {3,16,16}

  local fine8 = torch.FloatTensor(batchsize, 3, 8, 8)
  local real_fines32 = torch.Tensor(batchsize, 3, 32, 32)

  local nsamples = 30
  local distances = torch.Tensor(nsamples) --200

  for j = 1,nsamples do

    torch.setdefaulttensortype('torch.CudaTensor')

    xlua.progress(j, nsamples)

    for i = 1, batchsize do
      local sample = dataset[math.random(dataset:size())]
      cond_inputs1[i] = sample[1]:clone()  --load the label  10
      fine8[i] = sample[2]:clone():float()         --load the fine 3*8*8
      real_fines32[i] = sample[4]:clone()  --load the fines32 3*32*32
    end
    noise_inputs16:uniform(-1, 1)
    for i =1, batchsize do
      cond_inputs2[i] = image.scale(fine8[i], 16, 16):cuda()
    end

    local gen_differs16 = model_G16:forward({noise_inputs16, cond_inputs1, cond_inputs2})
    local gen_fines16 = torch.FloatTensor(batchsize, 3, 16, 16)
    -- add the 16 corase image
    for i =1,batchsize do
      gen_fines16[i] = torch.add(gen_differs16[i],1,cond_inputs2[i]):float()
    end
    -- upsample the fines16 , then gen the corases 32
    local gen_corases32 = torch.Tensor(batchsize, 3, 32, 32)
    for i =1 ,batchsize do
      gen_corases32[i]= image.scale(gen_fines16[i],32,32):cuda()
    end
    -- gen the differs32
    noise_inputs32:uniform(-1, 1)
    local gen_differs32 = model_G32:forward({noise_inputs32, cond_inputs1, gen_corases32})
    local gen_fines32 = torch.Tensor(batchsize, 3, 32, 32)
    -- add the 32 corase image
    for i =1,batchsize do
      gen_fines32[i] = torch.add(gen_differs32[i],1,gen_corases32[i])
    end

    -- compute distance
    local dist = 1e10
    for i = 1,batchsize do
      dist = math.min(torch.dist(real_fines32[i], gen_fines32[i]), dist) --128张图片和真图片的最小欧式距离
    end
    distances[j] = dist

    local to_plot = {}
    -- plot real fine32
     for i=1,batchsize do
      to_plot[i] = gen_differs16[i]:float()
    end
    local fname =  'sampling/2layers-gen_differs16-sample-' .. j .. '.png'
    torch.setdefaulttensortype('torch.FloatTensor')
    image.save(fname, image.toDisplayTensor{input=to_plot, scaleeach=true})

   for i=1,batchsize do
      to_plot[i] = gen_differs32[i]:float()
    end
    local fname =  'sampling/2layers-gen_differs32-sample-' .. j .. '.png'
    torch.setdefaulttensortype('torch.FloatTensor')
    image.save(fname, image.toDisplayTensor{input=to_plot, scaleeach=true})

     for i=1,batchsize do
      to_plot[i] = fine8[i]:float()
    end
    local fname =  'sampling/2layers-real_fines8-sample-' .. j .. '.png'
    torch.setdefaulttensortype('torch.FloatTensor')
    image.save(fname, image.toDisplayTensor{input=to_plot, scaleeach=true})

    for i=1,batchsize do
      to_plot[i] = gen_fines16[i]:float()
    end
    local fname =  'sampling/2layers-gen_fines16-sample-' .. j .. '.png'
    torch.setdefaulttensortype('torch.FloatTensor')
    image.save(fname, image.toDisplayTensor{input=to_plot, scaleeach=true})

    for i=1,batchsize do
      to_plot[i] = real_fines32[i]:float()
    end
    local fname =  'sampling/2layers-real_fines32-sample-' .. j .. '.png'
    torch.setdefaulttensortype('torch.FloatTensor')
    image.save(fname, image.toDisplayTensor{input=to_plot, scaleeach=true})

    for i=1,batchsize do
      to_plot[i] = gen_fines32[i]:float()
    end
    local fname =  'sampling/2layers-gen_fines32-sample-' .. j .. '.png'
    torch.setdefaulttensortype('torch.FloatTensor')
    image.save(fname, image.toDisplayTensor{input=to_plot, scaleeach=true})

  end
  print('average distance = ' .. distances:mean())
  print('min     distance = ' .. distances:min())
  print('max     distance = ' .. distances:max())
end

return getSamples16_32