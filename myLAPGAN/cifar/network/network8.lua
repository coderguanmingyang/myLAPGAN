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

local network8 = {}

function network8.getD()

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

      return model_D8
end

function network8.getG()

      local nplanes = opt.hidden_G8
      --用来存放输入数据：identity
      g_n_sc8 = nn.Identity()() -- noise (shaped as coarse map)
      g_c_sc8 = nn.Identity()() -- class vector

      g_class1_sc8 = nn.Linear(10, 8*8)(g_c_sc8) -- 10 inputs -> fine size * fine size outputs (default 16)
      g_class2_sc8 = nn.Reshape(1, 8, 8)(nn.ReLU()(g_class1_sc8)) --convert class vector into map
      -- 按列（第2维）
      g1_sc8 = nn.JoinTable(2, 2)({g_n_sc8, g_class2_sc8}) -- combine maps into 4 channels
      g2_sc8 = nn.SpatialConvolutionUpsample(5, nplanes, 3, 3, 1)(g1_sc8)   --filter nplanes*3*3
      g3_sc8 = nn.SpatialConvolutionUpsample(nplanes, nplanes, 3, 3, 1)(nn.ReLU()(g2_sc8))
      g4_sc8 = nn.SpatialConvolutionUpsample(nplanes, 3, 3, 3, 1)(nn.ReLU()(g3_sc8))
      model_G8 = nn.gModule({g_n_sc8, g_c_sc8}, {g4_sc8}) --生成一个有向无环图

      return model_G8
end

return network8
