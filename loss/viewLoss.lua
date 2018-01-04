local M = {}
require 'nn'
-------------------------------
-------------------------------
local function reverseCumsum(t, dim)
    -- o_k = \sum_{i=k}^{i^n} t_i
    local tSum = t:sum(dim)
    local szs = {}
    for d=1,t:nDimension() do
        szs[d] = 1
    end
    szs[dim] = t:size(dim)
    
    local tSumRep = tSum:repeatTensor(unpack(szs))
    local rSum = tSumRep - torch.cumsum(t, dim) + t
    return rSum
end

-------------------------------
-------------------------------
local RayEventProb, Parent = torch.class('nn.RayEventProb', 'nn.Module')

function RayEventProb:__init()
    Parent.__init(self)
end

-- x = B X 1 X W X H X D
-- returns qVals = B X 1 X W X H X (D+1)
-- Computed using Eq 2 from the DRC paper - https://arxiv.org/pdf/1704.06254.pdf
function RayEventProb:updateOutput(x)
    local B,_,W,H,D = unpack(torch.totable(x:size()))
    local onesInit = torch.ones(B, 1, W, H, 1):typeAs(x)
    local zerosEnd = torch.zeros(B, 1, W, H, 1):typeAs(x)
    
    local prod_i_minus_1 = torch.cumprod(torch.cat(onesInit, x, 5), 5)
    local prod_i = torch.cumprod(torch.cat(x, zerosEnd, 5), 5)
    
    self.output = prod_i_minus_1 - prod_i
    return self.output
end

-- x = B X 1 X W X H X D
-- psiVals = B X 1 X W X H X (D+1)
-- returns grad_x = B X 1 X W X H X D
-- Computed using Eq 7 from the DRC paper - https://arxiv.org/pdf/1704.06254.pdf
function RayEventProb:updateGradInput(x, psiVals)
    local B,_,W,H,D = unpack(torch.totable(x:size()))
    
    local psiDiff = psiVals:narrow(5,2,D) - psiVals:narrow(5,1,D)
    local prod_i = torch.cumprod(x, 5)
    local psiDiff_prod_i = torch.cmul(prod_i, psiDiff)
    
    local x_div_safe = torch.clamp(x, 1e-8, 1)
    local grad_x = torch.cdiv(reverseCumsum(psiDiff_prod_i, 5), x_div_safe)
    
    self.gradInput = grad_x
    return self.gradInput
end

-------------------------------
-------------------------------

local viewLoss = {}
viewLoss.__index = viewLoss

setmetatable(viewLoss, {
    __call = function (cls, ...)
        return cls.new(...)
    end,
})

function viewLoss.new(psiModule)
    local self = setmetatable({}, viewLoss)
    self.psiModule = psiModule
    self.qModule = nn.RayEventProb() -- event probability computer
    return self
end

-- pred = {x, [y]}; x = B X 1 X W X H X D, [y] = B X K_y X W X H X D
-- observations = B X K_o X W X H
-- depthSamples = D
function viewLoss:forwardBackward(pred, observations, depthSamples)
    local B,_,W,H,D = unpack(torch.totable(pred[1]:size()))
    local dSamplesRep = torch.repeatTensor(depthSamples,B,1,W,H,1)
    --print(observations:size(), B, W, H, D)
    local observations = observations:contiguous()
    local observationsRep = torch.repeatTensor(observations:view(B,-1,W,H,1),1,1,1,1,D)
    
    -- Forward Pass
    local qVals = self.qModule:forward(pred[1]) -- qVals = B X 1 X W X H X (D+1)
    
    self.qVals = qVals
    self.depthValsObs = dSamplesRep
    
    -- _, self.qMaxInds = qVals:max(5)
    --print(qMaxInds)
    --print(qVals:mean()*(D+1))
    local psiVals = self.psiModule:forward(pred[2], dSamplesRep, observationsRep) -- psiVals = B X 1 X W X H X (D+1)
    self.err_per_inst = torch.cmul(qVals, psiVals):sum(5):mean(4):mean(3)
    --self.err_per_pix = torch.cmul(qVals, psiVals):sum(5)
    local err = self.err_per_inst:mean()
    
    -- Backward Pass
    local denom = 1/(B*W*H)
    local grad_x = self.qModule:backward(pred[1], torch.mul(psiVals, denom))
    local grad_y = self.psiModule:backward(pred[2], dSamplesRep, observationsRep)
    -- Computed using Eq 8 from the DRC paper - https://arxiv.org/pdf/1704.06254.pdf
    if(grad_y) then
        local qValsNorm = torch.mul(qVals:narrow(5,1,D), denom)
        for c=1,grad_y:size(2) do
            --print(grad_y:size(), qVals:size())
            grad_y:narrow(2,c,1):cmul(qValsNorm)
        end
    end
    return err, {grad_x, grad_y}
end

function viewLoss:cuda()
    self.useCuda = true
    self.qModule = self.qModule:cuda()
    self.psiModule:cuda()
    return
end

-------------------------------
-------------------------------
M.reverseCumsum = reverseCumsum
M.qComputer = qComputer
M.viewLoss = viewLoss
return M