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

local getSamples8_16_32 = {}



function getSamples8_16_32.genImage( G8, G16, G32)

  torch.manualSeed(8)
  torch.setdefaulttensortype('torch.CudaTensor')
  --load the network
  local tmp8 = torch.load(G8)
  print('G8 load succsefully')
  local tmp16 = torch.load(G16)
  print('G16 load succsefully')
  local tmp32 = torch.load(G32)
  print('G32 load succsefully')
  local model_G8  = tmp8.G
  local model_G16 = tmp16.G
  local model_G32 = tmp32.G
  -- noise input
  local noise_inputs8
  noise_inputs8 = torch.Tensor(batchsize, noiseDim8[1], noiseDim8[2], noiseDim8[3]) --128*3*8*8
  local noise_inputs16
  noise_inputs16 = torch.Tensor(batchsize, noiseDim16[1], noiseDim16[2], noiseDim16[3]) --128*1*16*16
  local noise_inputs32
  noise_inputs32 = torch.Tensor(batchsize, noiseDim32[1], noiseDim32[2], noiseDim32[3]) --128*1*32*32
  -- conditon inputs
  local class
  class = torch.Tensor(batchsize, 10)  --default：conDim1 = 10
  -- tensor init
  --local real_fines32 = torch.FloatTensor(batchsize, 3, 32, 32)
  --local gen_I2 = torch.FloatTensor(batchsize, 3, 8, 8)
  local nsamples = 1
  -- 10 class
  for i =1 ,10 do
    --init class
    class:zero()
    for z =1,batchsize do
    class[z][i]=1
    end
    print('class ['..i..']\n')
    for j = 1,nsamples do

      xlua.progress(j, nsamples)
      torch.setdefaulttensortype('torch.CudaTensor')
      --torch.manualSeed(8)

      noise_inputs8:uniform(-1, 1)
      if i==1 and j==1 then
        print(noise_inputs8[1][1])
      end
      noise_inputs16:uniform(-1, 1)
      noise_inputs32:uniform(-1, 1)
      -- 3th layer
      local gen_I2 = model_G8:forward({noise_inputs8, class}):float()
      local coarse1 = torch.Tensor(batchsize,3,16,16)
      for m = 1, batchsize do
        coarse1[m] = image.scale(gen_I2[m],16,16):cuda()
      end
      -- 2th layer
      local gen_h1 = model_G16:forward({noise_inputs16, class, coarse1})
      local gen_I1 = torch.add(coarse1, 1, gen_h1):float()

      local coarse0 = torch.Tensor(batchsize,3,32,32)
      for m = 1, batchsize do
        coarse0[m] = image.scale(gen_I1[m],32,32):cuda()
      end
      --1th layer
      local gen_h0 = model_G32:forward({noise_inputs32, class, coarse0})
      local gen_I0 = torch.add(coarse0, 1, gen_h0)

      local to_plot = {}
      -- plot real fine32

      for i=1,batchsize do
        to_plot[i] = gen_I0[i]:float()
      end
      local fname =  'sampling/test_up/class-' .. i ..'-I0-'..'-samples-'..j.. '.png'
      torch.setdefaulttensortype('torch.FloatTensor')
      image.save(fname, image.toDisplayTensor{input=to_plot, scaleeach=true})

       for i=1,batchsize do
        to_plot[i] = gen_I1[i]:float()
      end
      local fname =  'sampling/test_up/class-' .. i..'-I1-'..'-samples-'..j.. '.png'
      torch.setdefaulttensortype('torch.FloatTensor')
      image.save(fname, image.toDisplayTensor{input=to_plot, scaleeach=true})

       for i=1,batchsize do
        to_plot[i] = gen_I2[i]:float()
      end
      local fname =  'sampling/test_up/class-' .. i ..'-I2-'..'-samples-'..j.. '.png'
      torch.setdefaulttensortype('torch.FloatTensor')
      image.save(fname, image.toDisplayTensor{input=to_plot, scaleeach=true})
    end
  end
end

return getSamples8_16_32