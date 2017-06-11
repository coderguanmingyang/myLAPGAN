require 'torch'
require 'paths'
require 'image'
image_utils = require 'utils.image'

cifar_scale32 = {}

cifar_scale32.path_dataset = 'cifar-10-batches-t7/'

cifar_scale32.coarseSize = 16
cifar_scale32.fineSize = 32

function cifar_scale32.init(fineSize, coarseSize)
  cifar_scale32.fineSize = fineSize
  cifar_scale32.coarseSize = coarseSize
end

function cifar_scale32.loadTrainSet(start, stop, augment, crop)
   return cifar_scale32.loadDataset(true, start, stop, augment, crop)
end

function cifar_scale32.loadTestSet(crop)
   return cifar_scale32.loadDataset(false, nil, nil, nil, crop)
end

function cifar_scale32.loadDataset(isTrain, start, stop, augment, crop)
  local data
  local labels
  local defaultType = torch.getdefaulttensortype()
  -- load train data
  if isTrain then
    data = torch.FloatTensor(50000, 3, 32, 32)
    labels = torch.FloatTensor(50000)
    for i = 0,4 do
      local subset = torch.load(cifar_scale32.path_dataset .. 'data_batch_' .. (i+1) .. '.t7', 'ascii')
      data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t():reshape(10000, 3, 32, 32)
      labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
    end
  else -- load test data
    subset = torch.load(cifar_scale32.path_dataset .. 'test_batch.t7', 'ascii')
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
  print('<cifar10><scale-32> loaded ' .. N .. ' examples')

  local dataset = {}
  dataset.data = data -- on cpu
  dataset.labels = labels
  --print(type(dataset.data))
  --print(dataset.data:size(1),dataset.data:size(2),dataset.data:size(3),dataset.data:size(4))
  dataset.coarseData = torch.FloatTensor(N, 3, cifar_scale32.fineSize, cifar_scale32.fineSize) --N * 3 * 32 * 32
  dataset.fineData = torch.FloatTensor(N, 3, cifar_scale32.fineSize, cifar_scale32.fineSize)
  dataset.diffData = torch.FloatTensor(N, 3, cifar_scale32.fineSize, cifar_scale32.fineSize)

  -- Coarse data: first downsampling ,then upsampling
  function dataset:makeCoarse()
    for i = 1,N do
      local tmp = image.scale(self.data[i], cifar_scale32.coarseSize, cifar_scale32.coarseSize) -- Rescale the height and width. default:Bilinear interpolation
      self.coarseData[i] = image.scale(tmp, cifar_scale32.fineSize, cifar_scale32.fineSize)
    end
  end

  -- Fine data
  function dataset:makeFine()
    for i = 1,N do
      self.fineData[i] = image.scale(self.data[i], cifar_scale32.fineSize, cifar_scale32.fineSize)
    end
  end

  -- Diff (coarse - fine)
  function dataset:makeDiff()
    for i=1,N do
      self.diffData[i] = torch.add(self.fineData[i], -1, self.coarseData[i])
    end
  end

  function dataset:size()
    return N
  end

  function dataset:numClasses()
    return 10
  end

  local labelvector = torch.zeros(10)

  setmetatable(dataset, {__index = function(self, index)
     local diff = self.diffData[index]
     local cond = self.coarseData[index]
     local fine = self.fineData[index]
     labelvector:zero()
     labelvector[self.labels[index]] = 1
     local example = {diff, labelvector, cond, fine}
     return example
end})

  return dataset
end
