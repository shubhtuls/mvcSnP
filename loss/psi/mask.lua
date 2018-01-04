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

function psiFunc.new(bgWt)
    local self = setmetatable({}, psiFunc)
    self.bgWt = bgWt
    return self
end

-- inputs
-- yPred : B X K_y X W X H X D, or None
-- depthVals : B X 1 X W X H X D
-- observations : B X K_o X W X H X D (with same value repeated across the D dimension)
-- observations are binary values : 1 == background, 0 == foreground

-- output
-- psiVals : B X 1 X W X H X (D+1)
function psiFunc:forward(yPred, depthVals, observations)
    local B,_,W,H,D = unpack(torch.totable(depthVals:size()))
    
    -- Eq 4 from the DRC paper https://arxiv.org/pdf/1704.06254.pdf
    local psiVals = observations:clone()
    local psiValsEscape = (1-observations:narrow(5,1,1)):clone()
    psiVals = torch.cat(psiVals, psiValsEscape,5)
    if(self.bgWt) then
        local multWt = 1-observations:clone()
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