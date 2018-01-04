require 'optim'
require 'cunn'
local M = {}

-------------------------------
-------------------------------
local AdvCriterion = {}
AdvCriterion.__index = AdvCriterion

setmetatable(AdvCriterion, {
    __call = function (cls, ...)
        return cls.new(...)
    end,
})

-- dataSampler:forward() gives target domain samples
-- netD is the adversary
function AdvCriterion.new(dataSampler, netD, batchSize)
    local self = setmetatable({}, AdvCriterion)
    self.dataSampler = dataSampler
    self.netD = netD
    self.useGpu = false --will be set to true if self:cuda() is called
    self.optimStateD = {
       learningRate = 0.0001,
       beta1 = 0.9,
    }
    self.criterion = nn.BCECriterion()
    self.real_label = 1
    self.fake_label = 0
    self.label = torch.Tensor(batchSize)
    return self
end

function AdvCriterion:cuda()
    self.useGpu = true
    self.netD = self.netD:cuda()
    self.criterion = self.criterion:cuda()
    self.label = self.label:cuda()
end

function AdvCriterion:optimInit()
    self.parametersD, self.gradParametersD = self.netD:getParameters()
end

function AdvCriterion:fdX(x, lossWt)
    self.gradParametersD:zero()

    -- train with real
    local real, _ = unpack(self.dataSampler:forward())
    --print(real:size())
    self.realSample = real:clone()
    if(self.useGpu) then real = real:cuda() end
    self.label:fill(self.real_label)

    local output = self.netD:forward(real)
    local errD_real = self.criterion:forward(output, self.label)
    local df_do = self.criterion:backward(output, self.label)
    self.netD:backward(real, lossWt*df_do)

    local fake = self.generatedSamples
    --print(fake:size())
    local output = self.netD:forward(fake)
    self.label:fill(self.fake_label)
    local errD_fake = self.criterion:forward(output, self.label)
    local df_do = self.criterion:backward(output, self.label)
    self.netD:backward(fake, lossWt*df_do)

    self.errD = errD_real + errD_fake
    return self.errD, self.gradParametersD
end

-- one step of training the adversary
function AdvCriterion:updateAdversary(lossWt)
    local fDx = function (x) return self:fdX(x, lossWt) end
    optim.adam(fDx, self.parametersD, self.optimStateD)
end

-- computes gradients for input wrt adversary
function AdvCriterion:updateGradInput(generatedSamples)
    self.generatedSamples = generatedSamples:clone()
    self.label:fill(self.real_label) -- fake labels are real for generator cost
    local output = self.netD:forward(generatedSamples)
    self.errG = self.criterion:forward(output, self.label)
    self.errG_perInst = torch.log(output):mul(-1)
    local df_do = self.criterion:backward(output, self.label)
    local df_dg = self.netD:updateGradInput(generatedSamples, df_do)
    return df_dg
end
-------------------------------
-------------------------------
M.AdvCriterion = AdvCriterion
return M