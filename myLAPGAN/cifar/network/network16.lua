--
-- Created by IntelliJ IDEA.
-- User: PC
-- Date: 2017/6/11
-- Time: 14:22
-- To change this template use File | Settings | File Templates.
--
require 'torch'
require 'optim'
require 'pl'
require 'paths'
require 'image'

local network16 = {}

function network16.getD()

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

      return model_D16
end

function network16.getG()

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

      return model_G16
end

return network16
