local M = {}
local transforms = dofile('../geometry/transforms.lua')
local gridUtil = dofile('../geometry/grid.lua')
local vLoss = dofile('../loss/viewLoss.lua')
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

local dprLoss = {}
dprLoss.__index = dprLoss

setmetatable(dprLoss, {
    __call = function (cls, ...)
        return cls.new(...)
    end,
})

function dprLoss.new(psiModule, nP)
    local self = setmetatable({}, dprLoss)
    -- the CNN predicts occupancies, the samplerX converts them to emptiness probabilities after sampling
    -- we didnt make the CNN predict emptiness probabilities because Trilinear sampling assigns '0' to points outside the grid.
    self.samplerX = nn.Sequential():add(sampler3d()):add(nn.MulConstant(-1)):add(nn.AddConstant(1))
    self.samplerY = sampler3d()
    self.viewLoss = vLoss.viewLoss(psiModule)
    self.transformModule = transforms.rigidTransform(nP)
    
    return self
end

-- shape : {x_bar,[y_bar]}. x_bar = B X 1 X Gx X Gy X Gz, y_bar = B X K_y X Gx X Gy X Gz
-- pose : {trans,quat}. trans : B X 3, rot : B X 4
-- observation : B X K_o X W X H
-- camInfo : {depthSamples, Ks}, Ks : B X 3 X 3, depthSamples : D
-- returns mean loss and gradients w.r.t shape, pose
-- assumes pose etc. w.r.t a canonical 0 centered grid between [-1, 1]
function dprLoss:forwardBackward(shape, pose, observation, camInfo)
    local tmrPre, tmrForw, tmrBack, tmrSampler, tmrTransform
    
    --tmrPre = torch.Timer(); tmrPre:reset();
    local ptsCam = self:computePtsCamCoord(observation:size(), camInfo)
    --self.ptsCam = ptsCam
    local B, W, H, D = unpack(torch.totable(ptsCam:size()))
    local nP = W*H*D
    --print(B,W,H,D)
    --tmrPre:stop()
    
    -- Forward Pass
    --tmrForw = torch.Timer(); tmrForw:reset()
    --tmrTransform = torch.Timer(); tmrTransform:reset()
    local ptsTransformed = self.transformModule:forward({ptsCam:view(B,-1,3), pose[1], pose[2]})
    --tmrTransform:stop();
    --print(ptsTransformed:min(2):min(1))
    --print(ptsCam:view(B,-1,3):mean(2):mean(1))
    
    ptsTransformed = ptsTransformed:view(B, W, H, D, 3)
    local predsCam = {}
    predsCam[1] = self.samplerX:forward({shape[1], ptsTransformed})
    --print(predsCam[1]:mean())
    if(#shape > 1) then
        predsCam[2] = self.samplerY:forward({shape[2], ptsTransformed})
    end
    --self.predsCam = predsCam
    
    local depthSamples = camInfo[1]
    if(self.useCuda) then depthSamples = depthSamples:cuda() end

    local err, gradPredsCam = self.viewLoss:forwardBackward(predsCam, observation, depthSamples)
    --tmrForw:stop()
    
    -- Backward Pass
    --tmrBack = torch.Timer(); tmrBack:reset();
    local gradSamplerX = self.samplerX:backward({shape[1], ptsTransformed}, gradPredsCam[1])
    local gradShape = {gradSamplerX[1]}
    local gradPtsTransformed = gradSamplerX[2]
    if (#shape > 1) then
        local gradSamplerY = self.samplerY:backward({shape[2], ptsTransformed}, gradPredsCam[2])
        gradShape[2] = gradSamplerY[1]
        gradPtsTransformed:add(gradSamplerY[2])
    end
    --tmrTransform:resume();
    local gradTransformer = self.transformModule:backward({ptsCam:view(B,-1,3), pose[1], pose[2]}, gradPtsTransformed)
    local gradPose = {gradTransformer[2], gradTransformer[3]}
    --tmrTransform:stop()
    --tmrBack:stop()
    ----
    --print(tmrPre:time().real, tmrForw:time().real, tmrBack:time().real, tmrTransform:time().real)
    return err, gradShape, gradPose
end

function dprLoss:computePtsCamCoord(oSize, camInfo)
    local depthSamples = camInfo[1]
    local Ks = camInfo[2]
    
    local B, _, W, H = unpack(torch.totable(oSize))
    local D = depthSamples:size(1)
    local ptsCam = gridUtil.meshGrid(torch.linspace(0.5,W-0.5,W),torch.linspace(0.5,H-0.5,H),torch.ones(D))
    ptsCam = torch.repeatTensor(ptsCam,B,1,1,1,1) -- B X W X H X D X 3
    --print(B,W,H,D,Ks:size(),ptsCam:size())
    for b=1,B do
        ptsCam[b]:narrow(4,1,1):add(-Ks[b][1][3]):div(Ks[b][1][1])
        ptsCam[b]:narrow(4,2,1):add(-Ks[b][2][3]):div(Ks[b][2][2])
    end
    for d=1,D do
        ptsCam:narrow(4,d,1):mul(depthSamples[d])
    end
    if(self.useCuda) then
        ptsCam = ptsCam:cuda()
    end
    return ptsCam
end

function dprLoss:cuda()
    self.useCuda = true
    self.transformModule = self.transformModule:cuda()
    self.samplerX = self.samplerX:cuda()
    self.samplerY = self.samplerY:cuda()
    self.viewLoss:cuda()
    return
end

-------------------------------
-------------------------------
M.dprLoss = dprLoss
return M