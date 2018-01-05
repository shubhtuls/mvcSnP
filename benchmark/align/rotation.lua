local M = {}
local quatUtil = dofile('../geometry/quatUtils.lua')
dofile('../loss/quatLoss.lua')
require 'nn'
require 'nngraph'
-------------------------------
-------------------------------
local quatAlign = {}
quatAlign.__index = quatAlign

setmetatable(quatAlign, {
    __call = function (cls, ...)
        return cls.new(...)
    end,
})

function quatAlign.new()
    local self = setmetatable({}, quatAlign)
    -- the CNN predicts occupancies, the samplerX converts them to emptiness probabilities after sampling
    -- we didnt make the CNN predict emptiness probabilities because Trilinear sampling assigns '0' to points outside the grid.
    self.errModule = nn.QuatCriterion()
    
    local predQuat = - nn.Identity()
    local transformQuat = - nn.Identity()
    
    local pred = predQuat - nn.Unsqueeze(2)
    local transform = transformQuat - nn.Unsqueeze(2)
    local transformed = {transform, pred} - quatUtil.HamiltonProductModule() - nn.Sum(2)
    local gmod = nn.gModule({predQuat, transformQuat}, {transformed})
    
    self.transformModule = gmod
    return self
end

function quatAlign:align(pred, gt, numRestarts, numIter)
    local numRestarts = numRestarts or 50
    local numIter = numIter or 300
    local stepSize = 10
    
    local bestQuat = torch.Tensor(4):typeAs(pred)
    local errMin = 10
    
    for rs = 1,numRestarts do
        print(rs, errMin)
        local quatInit = torch.Tensor(4):uniform(-1,1):typeAs(pred)
        quatInit:div(quatInit:norm())
        
        local quat = quatInit:clone()
        for ix=1,numIter do
            local grad, err = self:getGradient(pred, gt, quat)
            quat:add(grad:clone():mul(-stepSize))
            quat:div(quat:norm())
            
            if(err < errMin) then
                errMin = err
                bestQuat:copy(quat)
            end
        end
    end
    
    return bestQuat
end

function quatAlign:transform(pred, pose)
    local B = pred:size(1)
    local pose = torch.repeatTensor(pose,B,1)
    local predTransformed = self.transformModule:forward({pred, pose})
    return predTransformed
end

-- Pred (and gt) is B X 4
-- Pose is a quaternion
function quatAlign:getGradient(pred, gt, pose)
    local B = pred:size(1)
    local pose = torch.repeatTensor(pose,B,1)
    local predTransformed = self.transformModule:forward({pred, pose})
    
    local err = self.errModule:forward(predTransformed, gt)
    --print(err)
    
    local gradPredTransformed = self.errModule:backward(predTransformed, gt)
    -- transformModule left multiplies R_pose to R_pred
    local gradTransformer = self.transformModule:backward({pred, pose}, gradPredTransformed)
    local gradPose = gradTransformer[2]:mean(1)
    return gradPose, err
end

function quatAlign:cuda()
    self.useCuda = true
    self.transformModule = self.transformModule:cuda()
    self.errModule = self.errModule:cuda()
    return
end

-------------------------------
-------------------------------
M.quatAlign = quatAlign
return M