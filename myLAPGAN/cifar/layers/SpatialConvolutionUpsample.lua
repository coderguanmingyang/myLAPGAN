local SpatialConvolutionUpsample, parent = torch.class('nn.SpatialConvolutionUpsample','nn.SpatialConvolution')


-- module = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH])
-- [
-- nInputPlane:  The number of expected input planes in the image given into forward().
-- nOutputPlane: The number of output planes the convolution layer will produce
-- kW: The kernel width of the convolution
-- kH: The kernel height of the convolution
-- dW: The step of the convolution in the width dimension. Default is 1.
-- dH: The step of the convolution in the height dimension. Default is 1.
-- padW: The additional zeros added per width to the input planes. Default is 0, a good number is (kW-1)/2.
-- padH: The additional zeros added per height to the input planes. Default is padW, a good number is (kH-1)/2
-- ]
-- 1.If the input image is a 3D tensor nInputPlane x height x width,
-- the output image size will be  nOutputPlane x owidth x oheight where:
-- owidth  = (width  - kW) / dW + 1
-- oheight = (height - kH) / dH + 1
-- 2.如果pad不为0的话：
-- owidth  = floor((width  + 2*padW - kW) / dW + 1)
-- oheight = floor((height + 2*padH - kH) / dH + 1)
function SpatialConvolutionUpsample:__init(nInputPlane, nOutputPlane, kW, kH, factor)
   factor = factor or 2
   assert(kW and kH and nInputPlane and nOutputPlane)
   assert(kW % 2 == 1, 'kW has to be odd')
   assert(kH % 2 == 1, 'kH has to be odd')
   self.factor = factor 
   self.kW = kW
   self.kH = kH
   self.nInputPlaneU = nInputPlane
   self.nOutputPlaneU = nOutputPlane
   parent.__init(self, nInputPlane, nOutputPlane * factor * factor, kW, kH, 1, 1, (kW-1)/2)
   -- dw,dh =1 padw,padh=(kW-1)/2 ,so the hight and width is not changed
end

function SpatialConvolutionUpsample:updateOutput(input)
   self.output = parent.updateOutput(self, input)
   if input:dim() == 4 then
      self.h = input:size(3)
      self.w = input:size(4)
      self.output = self.output:view(input:size(1), self.nOutputPlaneU, self.h*self.factor, self.w*self.factor)
   else
      self.h = input:size(2)
      self.w = input:size(3)
      self.output = self.output:view(self.nOutputPlaneU, self.h*self.factor, self.w*self.factor)
   end
   return self.output
end

function SpatialConvolutionUpsample:updateGradInput(input, gradOutput)
   if input:dim() == 4 then
      gradOutput = gradOutput:view(input:size(1), self.nOutputPlaneU*self.factor*self.factor, self.h, self.w)
   else
      gradOutput = gradOutput:view(self.nOutputPlaneU*self.factor*self.factor, self.h, self.w)
   end
   self.gradInput = parent.updateGradInput(self, input, gradOutput)
   return self.gradInput
end

function SpatialConvolutionUpsample:accGradParameters(input, gradOutput, scale)
   if input:dim() == 4 then
      gradOutput = gradOutput:view(input:size(1), self.nOutputPlaneU*self.factor*self.factor, self.h, self.w)
   else
      gradOutput = gradOutput:view(self.nOutputPlaneU*self.factor*self.factor, self.h, self.w)
   end
   parent.accGradParameters(self, input, gradOutput, scale)
end

function SpatialConvolutionUpsample:accUpdateGradParameters(input, gradOutput, scale)
   if input:dim() == 4 then
      gradOutput = gradOutput:view(input:size(1), self.nOutputPlaneU*self.factor*self.factor, self.h, self.w)
   else
      gradOutput = gradOutput:view(self.nOutputPlaneU*self.factor*self.factor, self.h, self.w)
   end
   parent.accUpdateGradParameters(self, input, gradOutput, scale)
end
