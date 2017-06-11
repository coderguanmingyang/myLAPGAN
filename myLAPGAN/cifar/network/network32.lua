--
-- Created by IntelliJ IDEA.
-- User: PC
-- Date: 2017/6/11
-- Time: 14:30
-- To change this template use File | Settings | File Templates.
--

require 'torch'
require 'optim'
require 'pl'
require 'paths'
require 'image'

local network32 = {}

function network32.getD()

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

      return model_D32
end

function network32.getG()

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

      return model_G32
end

return network32

