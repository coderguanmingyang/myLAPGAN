require 'torch'
require 'paths'
require 'image'
image_utils = require 'utils.image'

cifar_scale8 = {}

cifar_scale8.path_dataset = 'cifar-10-batches-t7/'

--cifar_scale8.coarseSize = 8
cifar_scale8.fineSize = 8

function cifar_scale8.init(fineSize)
  cifar_scale8.fineSize = fineSize
end

function cifar_scale8.loadTrainSet(start, stop, augment, crop)
   return cifar_scale8.loadDataset(true, start, stop, augment, crop)
end

function cifar_scale8.loadTestSet(crop)
   return cifar_scale8.loadDataset(false, nil, nil, nil, crop)
end

function cifar_scale8.loadDataset(isTrain, start, stop, augment, crop)
  local data
  local labels
  local defaultType = torch.getdefaulttensortype()
  --print(defaultType)

  -- load train data
  if isTrain then
    data = torch.FloatTensor(50000, 3, 32, 32)
    labels = torch.FloatTensor(50000)
    for i = 0,4 do
      local subset = torch.load(cifar_scale8.path_dataset .. 'data_batch_' .. (i+1) .. '.t7', 'ascii')
      data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t():reshape(10000, 3, 32, 32)
      labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
    end
  else -- load test data
    subset = torch.load(cifar_scale8.path_dataset .. 'test_batch.t7', 'ascii')
    data = subset.data:t():reshape(10000, 3, 32, 32):type('torch.FloatTensor')
    labels = subset.labels:t():type(defaultType)
  end

  local start = start or 1
  local stop = stop or data:size(1) --data 的第一维的尺寸

  -- select chunk
  data = data[{ {start, stop} }]
  labels = labels[{ {start, stop} }]
  labels:add(1) -- 所有元素+1  becasue indexing is 1-based

  local N = stop - start + 1
  print('<cifar10><scale-8> loaded ' .. N .. ' examples')

  local dataset = {}  -- a table
  dataset.data = data -- on cpu
  dataset.labels = labels

  dataset.fineData = torch.FloatTensor(N, 3, cifar_scale8.fineSize, cifar_scale8.fineSize)   --N * 3 * 8 * 8
  dataset.fineData32 = torch.FloatTensor(N, 3, 32, 32)
  dataset.labelvectors = torch.FloatTensor(N, 10):fill(0)   --N * 10

  for i=1,N do
    dataset.labelvectors[i][dataset.labels[i]] = 1
    dataset.fineData32[i] = data[i]:clone()
  end

  -- fine data: first downsampling ,then downsampling 32->16->8
  function dataset:makeFine()
    for i = 1,N do
       -- Rescale the height and width. default:Bilinear interpolation
      local tmp = image.scale(self.data[i], 16, 16)
      self.fineData[i] = image.scale(tmp, cifar_scale8.fineSize, cifar_scale8.fineSize) --N * 3 * 8 * 8
    end
  end

  function dataset:size()
    return N
  end

  function dataset:numClasses()
    return 10
  end

  --local labelvector = torch.Tensor(10)
  --local labelvector = torch.zeros(10)
  --print('\nlabelvector '..type(labelvector)..'\n')

  setmetatable(dataset, {__index = function(self, index)

     local labelvector = self.labelvectors[index]
     local fine8 = self.fineData[index]
     local class = self.labels[index]
     local fine32 = self.fineData32[index]
     local example = {labelvector, fine8, class, fine32}
     return example
end})

  return dataset
end
