local M = {}
local transforms = dofile('../geometry/transforms.lua')
local gridUtil = dofile('../geometry/grid.lua')
require 'nn'
require 'nngraph'
require 'stn3d'
-------------------------------
-------------------------------
local function sampler3d()
    local x = -nn.Identity()
    local grid = -nn.Identity()
    
    local xPerm = x - nn.Transpose({2,3},{3,4},{4,5})
    local xSampled = {xPerm, grid} - nn.TrilinearSamplerBTHWC()
    local xSampledUnperm = xSampled - nn.Transpose({4,5},{3,4},{2,3})
    
    local gmod = nn.gModule({x, grid}, {xSampledUnperm})
    return gmod
end
-------------------------------
-------------------------------
local shapeAlign = {}
shapeAlign.__index = shapeAlign

setmetatable(shapeAlign, {
    __call = function (cls, ...)
        return cls.new(...)
    end,
})

function shapeAlign.new(errModule, gridSize)
    local self = setmetatable({}, shapeAlign)
    -- the CNN predicts occupancies, the samplerX converts them to emptiness probabilities after sampling
    -- we didnt make the CNN predict emptiness probabilities because Trilinear sampling assigns '0' to points outside the grid.
    self.sampler = sampler3d()
    self.errModule = errModule
    --self.transformModule = transforms.rotation(gridSize[1]*gridSize[2]*gridSize[3])
    self.transformModule = transforms.rigidTransform(gridSize[1]*gridSize[2]*gridSize[3])
    return self
end

-- returns the transformation to go from GT frame to predicted frame
function shapeAlign:align(pred, gt, numRestarts, numIter)
    local numRestarts = numRestarts or 50
    local numIter = numIter or 300
    local stepSize = 1
    
    local bestQuat = torch.Tensor(4):typeAs(pred)
    local bestTrans = torch.Tensor(3):typeAs(pred):fill(0)
    local errMin = 1
    local transInit = torch.Tensor(3):fill(0):typeAs(pred)
    
    for rs = 1,numRestarts do
        --print(rs, errMin)
        local quatInit = torch.Tensor(4):uniform(-1,1):typeAs(pred)
        quatInit:div(quatInit:norm())
        
        local quat = quatInit:clone()
        local trans = transInit:clone()
        local err_final
        for ix=1,numIter do
            local grad, err = self:getGradient(pred, gt, {trans, quat})
            quat:add(grad[2]:clone():mul(-stepSize))
            quat:div(quat:norm())
            trans:add(grad[1]:clone():mul(-stepSize))
            if ix==numIter then
                err_final = err
            end
        end
        if(err_final < errMin) and (trans:abs():max() < 0.2) then
            errMin = err_final
            bestQuat:copy(quat)
            bestTrans:copy(trans)
            --print(err_final, quat, trans)
        end
    end
    
    self:getGradient(pred, gt, {bestTrans, bestQuat}) --need a forward pass through sampler
    return {bestTrans, bestQuat}, self.sampler.output:clone()

end

function shapeAlign:transform(pred, pose)
    local ptsSample = self:computePtsSample(pred:size())
    local B, W, H, D = unpack(torch.totable(ptsSample:size()))
    local quat, trans
    if (not torch.isTensor(pose)) then
        trans = torch.repeatTensor(pose[1],B,1)
        quat = torch.repeatTensor(pose[2],B,1)
    else
        trans = torch.Tensor(B,3):typeAs(pose):fill(0)
        quat = torch.repeatTensor(pose,B,1)
    end
    local ptsTransformed = self.transformModule:forward({ptsSample:view(B,-1,3), trans, quat})
    
    ptsTransformed = ptsTransformed:view(B, W, H, D, 3)
    local predTransformed = self.sampler:forward({pred, ptsTransformed})
    return predTransformed
end

-- Pred (and gt) is B X 1 X H X W X D
-- Pose is a quaternion
function shapeAlign:getGradient(pred, gt, pose)
    local ptsSample = self:computePtsSample(pred:size())
    local B, W, H, D = unpack(torch.totable(ptsSample:size()))

    local quat, trans
    if (not torch.isTensor(pose)) then
        trans = torch.repeatTensor(pose[1],B,1)
        quat = torch.repeatTensor(pose[2],B,1)
    else
        trans = torch.Tensor(B,3):typeAs(pose):fill(0)
        quat = torch.repeatTensor(pose,B,1)
    end
    local ptsTransformed = self.transformModule:forward({ptsSample:view(B,-1,3), trans, quat})
    
    ptsTransformed = ptsTransformed:view(B, W, H, D, 3)
    local predTransformed = self.sampler:forward({pred, ptsTransformed})
    local err = self.errModule:forward(predTransformed, gt)
    --print(err)
    
    local gradPredTransformed = self.errModule:backward(predTransformed, gt)
    local gradSampler = self.sampler:backward({pred, ptsTransformed}, gradPredTransformed)
    
    local gradTransformer = self.transformModule:backward({ptsTransformed:view(B,-1,3), trans, quat}, gradSampler[2])
    local gradTrans = gradTransformer[2]:mean(1)
    local gradQuat = gradTransformer[3]:mean(1)
    return {gradTrans, gradQuat}, err
end

function shapeAlign:computePtsSample(predSize)
    local B, _, W, H, D = unpack(torch.totable(predSize))
    local ptsSample = gridUtil.meshGrid(torch.linspace(-W+0.5,W-0.5,W):div(W),torch.linspace(-H+0.5,H-0.5,H):div(H),torch.linspace(-D+0.5,D-0.5,D):div(D))
    ptsSample = torch.repeatTensor(ptsSample,B,1,1,1,1) -- B X W X H X D X 3
    if(self.useCuda) then ptsSample = ptsSample:cuda() end
    return ptsSample
end

function shapeAlign:cuda()
    self.useCuda = true
    self.transformModule = self.transformModule:cuda()
    self.sampler = self.sampler:cuda()
    self.errModule = self.errModule:cuda()
    return
end

-------------------------------
-------------------------------
M.shapeAlign = shapeAlign
return M