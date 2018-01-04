local M = {}
-------------------------------
-------------------------------
local psiFunc = {}
psiFunc.__index = psiFunc

setmetatable(psiFunc, {
    __call = function (cls, ...)
        return cls.new(...)
    end,
})

function psiFunc.new(maxDepth, bgWt)
    local self = setmetatable({}, psiFunc)
    self.maxDepth = maxDepth
    self.bgWt = bgWt
    return self
end

-- inputs
-- yPred : B X K_y X W X H X D, or None
-- depthVals : B X 1 X W X H X D
-- observations : B X K_o X W X H X D (with same value repeated across the D dimension)

-- output
-- psiVals : B X 1 X W X H X (D+1)
function psiFunc:forward(yPred, depthVals, observations)
    local B,_,W,H,D = unpack(torch.totable(depthVals:size()))
    
    -- Eq 3 from the DRC paper https://arxiv.org/pdf/1704.06254.pdf
    local psiVals = (depthVals - observations):abs()
    local psiValsEscape = (self.maxDepth-observations:narrow(5,1,1)):abs()
    psiVals = torch.cat(psiVals, psiValsEscape,5)
    if(self.bgWt) then
        local multWt = torch.lt(observations, self.maxDepth - 1e-6):typeAs(observations)
        multWt = torch.cat(multWt, multWt:narrow(5,1,1):clone())
        multWt:mul(1-self.bgWt):add(self.bgWt)
        psiVals:cmul(multWt)
    end
    return psiVals
end

function psiFunc:backward(yPred, depthVals, observations)
    return nil
end

function psiFunc:cuda()
    return
end

-------------------------------
-------------------------------
M.psiFunc = psiFunc
return M