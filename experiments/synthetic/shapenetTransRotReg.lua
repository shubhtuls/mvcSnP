-- nPoses=8 useEuler=1 numDepthSamples=80 gpu=1 disp=1 visIter=100 numTrainIter=80000 obsType=depth synset=3001627 quatPriorWt=1 name=chair_depth_poseRegUnsup_nds80_np8_euler_prior th synthetic/shapenetPoseReg.lua
torch.manualSeed(1)
torch.setnumthreads(1) --code is slower if using more !!
require 'cunn'
require 'optim'
local data = dofile('../data/synthetic/shapenet.lua')
local netBlocks = dofile('../nnutils/netBlocks.lua')
local dprLoss = dofile('../loss/dprLoss.lua')
local netInit = dofile('../nnutils/netInit.lua')
local vUtil = dofile('../utils/visUtils.lua')
local splitUtil = dofile('../benchmark/synthetic/splits.lua')
local miscUtil = dofile('../utils/misc.lua')
local advLoss = dofile('../nnutils/advLoss.lua')
local quatUtil = dofile('../geometry/quatUtils.lua')
dofile('../loss/quatLoss.lua')
-----------------------------
--------parameters-----------
local gridBoundSnet = 0.5 --parameter fixed according to shapenet models' size. Do not change
local gridBoundStn = 1 --trilinear sampler assumes the grid represents a volume in [-1,1]. Do not change.
local extrinsicScale = gridBoundStn/gridBoundSnet --do not change
local bgDepth = extrinsicScale*10.0 --parameter fixed according to rendering used. Do not change.
local useCudaLoss = true
local defaultZ = extrinsicScale*2.0 -- models were rendered with this Z

local params = {}
--params.bgVal = 0
params.name = 'chair_depth_transPoseRegUnsup_nds80_prior'
params.gpu = 1
params.batchSize = 8
params.nImgs = 5
params.nCams = 5
params.imgSizeY = 64
params.imgSizeX = 64
params.obsSizeY = 32
params.obsSizeX = 32
params.numDepthSamples = 80

params.bgWt = 0.2 -- figured out via cross-validation on the val set. Code currently ignoring this.
params.synset = 3001627 --chair:3001627, aero:2691156, car:2958343

params.gridSizeX = 32
params.gridSizeY = 32
params.gridSizeZ = 32
params.useEuler = 1 -- 1 means we'll predict quaternion via euler angle prediction

params.imsave = 0
params.disp = 0
params.obsType = 'depth'
params.bottleneckSize = 100
params.visIter = 100
params.nConvEncLayers = 5
params.nConvDecLayers = 4
params.nConvEncChannelsInit = 8
params.numTrainIter = 100000
params.quatSupWt = 0
params.quatPriorWt = 1
params.nPoses = 1

params.elPredMin = -20
params.elPredRange = 60
params.elPriorMin = -20
params.elPriorRange = 60

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(params) do params[k] = tonumber(os.getenv(k)) or os.getenv(k) or params[k] end

if params.disp == 0 then params.display = false else params.display = true end
if params.imsave == 0 then params.imsave = false end
params.visDir = '../cachedir/visualization/shapenet/' .. params.name
params.snapshotDir = '../cachedir/snapshots/shapenet/' .. params.name
params.imgSize = torch.Tensor({params.imgSizeY, params.imgSizeX})
params.obsSize = torch.Tensor({params.obsSizeY, params.obsSizeX})
params.gridSize = torch.Tensor({params.gridSizeX, params.gridSizeY, params.gridSizeZ})
params.synset = '0' .. tostring(params.synset) --to resolve string/number issues in passing bash arguments
params.modelsDataDir = '../cachedir/blenderRenderPreprocessTrans/' .. params.synset .. '/'
params.elPredMin = params.elPredMin*math.pi/180
params.elPredRange = params.elPredRange*math.pi/180
print(params)
-----------------------------
-----------------------------
paths.mkdir(params.visDir)
paths.mkdir(params.snapshotDir)
cutorch.setDevice(params.gpu)
local fout = io.open(paths.concat(params.snapshotDir,'log.txt'), 'w')
for k,v in pairs(params) do
    fout:write(string.format('%s : %s\n',tostring(k),tostring(v)))
end
fout:flush()

-----------------------------
----------LossComp-----------
local psiFunc =  dofile('../loss/psi/' .. params.obsType .. '.lua')
local psiModule
if(params.obsType == 'mask') then
    psiModule = psiFunc.psiFunc(params.bgWt)
elseif(params.obsType == 'depth') then
    psiModule = psiFunc.psiFunc(bgDepth, params.bgWt)
end
local depthSamples = torch.linspace(1,3,params.numDepthSamples + 1):mul(extrinsicScale)
local lossFunc = dprLoss.dprLoss(psiModule, depthSamples:size(1)*params.obsSize[1]*params.obsSize[2])
local lossFuncQuat = nn.QuatCriterion()

if(useCudaLoss) then
    lossFunc:cuda()
end
lossFuncQuat = lossFuncQuat:cuda()
-----------------------------
-------Pose Prior Loss-------
local netPoseD = nn.Sequential():add(nn.Linear(4,10)):add(nn.LeakyReLU(0.2, true)):add(nn.Linear(10,10)):add(nn.LeakyReLU(0.2, true)):add(nn.Linear(10,1)):add(nn.Sigmoid())
netPoseD:apply(netInit.weightsInit)
local poseSampler = {}
function poseSampler.forward(ignoreVar)
    local quats = torch.Tensor(params.nCams*params.batchSize,4)
    local flipQ = torch.Tensor(params.nCams*params.batchSize):random(0,1)
    for ix=1,params.nCams*params.batchSize do
        local rot = quatUtil.azel2rot(torch.random(0,359) + torch.uniform(-1,1), params.elPriorMin + torch.random(0,params.elPriorRange-1) + torch.uniform(-1,1))
        local quat = quatUtil.rot2quat(torch.inverse(rot))
        quats[ix]:copy(quat)
        if(flipQ[ix]==1) then
            quats[ix]:mul(-1)
        end
    end
    return {quats}
end
local AdvCriterionPose = advLoss.AdvCriterion(poseSampler, netPoseD, params.nCams*params.batchSize)
AdvCriterionPose:cuda()
AdvCriterionPose:optimInit()

-----------------------------
----------Encoder------------
local encoder, nOutChannels = netBlocks.convEncoderSimple2d(params.nConvEncLayers,params.nConvEncChannelsInit,3,true) --output is nConvEncChannelsInit*pow(2,nConvEncLayers-1) X imgSize/pow(2,nConvEncLayers)
local featSpSize = params.imgSize/torch.pow(2,params.nConvEncLayers)
--print(featSpSize)
local bottleneck = nn.Sequential():add(nn.Reshape(nOutChannels*featSpSize[1]*featSpSize[2],1,1,true))
local nInputCh = nOutChannels*featSpSize[1]*featSpSize[2]
for nLayers=1,2 do --fc for joint reasoning
    bottleneck:add(nn.SpatialConvolution(nInputCh,params.bottleneckSize,1,1)):add(nn.SpatialBatchNormalization(params.bottleneckSize)):add(nn.LeakyReLU(0.2, true))
    nInputCh = params.bottleneckSize
end
encoder:add(bottleneck)
encoder:apply(netInit.weightsInit)
--print(encoder)
---------------------------------
----------World Decoder----------
local ncOut = 1
local featSpSize = params.gridSize/torch.pow(2,params.nConvDecLayers)
local decoder  = nn.Sequential():add(nn.SpatialConvolution(params.bottleneckSize,nOutChannels*featSpSize[1]*featSpSize[2]*featSpSize[3],1,1,1)):add(nn.SpatialBatchNormalization(nOutChannels*featSpSize[1]*featSpSize[2]*featSpSize[3])):add(nn.ReLU(true)):add(nn.Reshape(nOutChannels,featSpSize[1],featSpSize[2],featSpSize[3],true))
decoder:add(netBlocks.convDecoderSimple3d(params.nConvDecLayers,nOutChannels,params.nConvEncChannelsInit,ncOut,true))
decoder:apply(netInit.weightsInit)

-----------------------------
-------Encoder Pose----------
local encoderPose, nOutChannels = netBlocks.convEncoderSimple2d(params.nConvEncLayers,params.nConvEncChannelsInit,3,true) --output is nConvEncChannelsInit*pow(2,nConvEncLayers-1) X imgSize/pow(2,nConvEncLayers)
local featSpSize = params.imgSize/torch.pow(2,params.nConvEncLayers)
--print(featSpSize)
local bottleneckPose = nn.Sequential():add(nn.Reshape(nOutChannels*featSpSize[1]*featSpSize[2],1,1,true))
local nInputCh = nOutChannels*featSpSize[1]*featSpSize[2]
for nLayers=1,2 do --fc for joint reasoning
    bottleneckPose:add(nn.SpatialConvolution(nInputCh,params.bottleneckSize,1,1)):add(nn.SpatialBatchNormalization(params.bottleneckSize)):add(nn.LeakyReLU(0.2, true))
    nInputCh = params.bottleneckSize
end
encoderPose:add(bottleneckPose):add(nn.Reshape(params.bottleneckSize,true))

local rotPred = quatUtil.quatPredSampleModule(params.bottleneckSize,params.nPoses,params.useEuler == 1,params.elPredMin,params.elPredRange)
local transPred = nn.Sequential():add(nn.Linear(params.bottleneckSize, 3)):add(nn.Tanh()):add(nn.MulConstant(0.25))
encoderPose:add(nn.ConcatTable():add(transPred):add(rotPred))

encoderPose:apply(netInit.weightsInit)
encoderPose.modules[#encoderPose.modules]:apply(netInit.weightsInitPose)
--print(encoderPose)
--print(encoder)

-----------------------------
----------Recons-------------
local splitUtil = dofile('../benchmark/synthetic/splits.lua')
local trainModels = splitUtil.getSplit(params.synset, nil, params.modelsDataDir)['train']
local dataLoader = data.dataLoader(params.modelsDataDir, params.batchSize, params.nCams, params.nImgs, params.obsSize, params.imgSize, params.obsType, trainModels, extrinsicScale)
local netRecons = nn.Sequential():add(encoder):add(decoder)

netRecons = netRecons:cuda()
encoderPose = encoderPose:cuda()
local err, errQuat = 0, 0

-- Optimization parameters
local optimState = {
   learningRate = 0.0001,
   beta1 = 0.9,
}

local optimStatePose = {
   learningRate = 0.0001,
   beta1 = 0.9,
}

local netParameters, netGradParameters = netRecons:getParameters()
local netPoseParameters, netPoseGradParameters = encoderPose:getParameters()
local imgs, pred, observations, camKs, imgsObsColor, camData, gradPosePred, posePred, errPerInst
local loss_tm = torch.Timer()

-----------------------------
-------Training Func---------
-- fX required for training
local fx = function(x)
    netGradParameters:zero()
    imgs, observations, camKs, imgsObsColor, camData = dataLoader:forward()
    observations = observations:transpose(4,5):contiguous() -- observations were Height X Width, we want width X height for loss
    imgs = imgs:cuda()

    --pred = netRecons:forward(imgs)
    pred = netRecons:forward(imgs)
    if(torch.isTensor(pred)) then
        pred = {pred}
    end
    loss_tm:reset(); loss_tm:resume()
    imgsObsColor = imgsObsColor:reshape(params.nCams*params.batchSize,3,imgsObsColor:size(4),imgsObsColor:size(5)):cuda()
    posePred = encoderPose:forward(imgsObsColor)
    
    gradPosePred = {posePred[1]:clone():fill(0), posePred[2]:clone():fill(0)}
    errPerInst = torch.Tensor(params.nCams*params.batchSize):cuda()
    
    --print(camData[3])
    local transPredErr = posePred[1]:clone()
    transPredErr:narrow(2,3,1):add(defaultZ)
    print(transPredErr:clone():csub(camData[3]:cuda()):abs():mean(1))

    local gradPred
    err = 0
    for nc=1,observations:size(1) do
        --print(nc)
        local err_nc, gradPred_nc, gradPosePred_nc

        local trans_nc = posePred[1]:narrow(1,1+(nc-1)*params.batchSize,params.batchSize)
        trans_nc:narrow(2,3,1):add(defaultZ)
        
        local rot_nc = posePred[2]:narrow(1,1+(nc-1)*params.batchSize,params.batchSize)
        --print(pose_nc:size(), camData[2][nc]:size())
        err_nc, gradPred_nc, gradPosePred_nc = lossFunc:forwardBackward(pred, {trans_nc:contiguous(), rot_nc:contiguous()}, observations[nc]:cuda(), {depthSamples, camKs[nc]})
        
        errPerInst:narrow(1,1+(nc-1)*params.batchSize,params.batchSize):copy(lossFunc.viewLoss.err_per_inst)
        gradPred = (nc==1) and gradPred_nc or miscUtil.addTableRec(gradPred_nc, gradPred, 1, 1)
        
        gradPosePred[1]:narrow(1,1+(nc-1)*params.batchSize,params.batchSize):copy(gradPosePred_nc[1]:div(observations:size(1)))
        gradPosePred[2]:narrow(1,1+(nc-1)*params.batchSize,params.batchSize):copy(gradPosePred_nc[2]:div(observations:size(1)))
        err = err + err_nc
    end
    err = err/observations:size(1)
    for ix = 1,#gradPred do
        gradPred[ix]:div(observations:size(1))
        gradPred[ix]=gradPred[ix]:cuda()
    end
    if(#gradPred==1) then
        gradPred = gradPred[1]
    end
    print('Loss Time : ' .. loss_tm:time().real)
    netRecons:backward(imgs, gradPred)
    return err, netGradParameters
end

local fxPose = function(x)
    netPoseGradParameters:zero()
    if(params.quatSupWt > 0) then
        local rotGt = camData[2]:clone():reshape(params.nCams*params.batchSize,4):cuda()
        errQuat = lossFuncQuat:forward(posePred[2]:cuda(), rotGt)
        local gradQuat = lossFuncQuat:backward(posePred[2]:cuda(), poseGt)
        gradPosePred[2]:add(gradQuat:mul(params.quatSupWt))
    end
    
    -- Pose Prior Loss
    local flipQ = torch.Tensor(params.nCams*params.batchSize):random(0,1)    
    local poseGenSamples = posePred[2]:cuda():clone()
    for ix=1,params.nCams*params.batchSize do
        if(flipQ[ix]==1) then
            poseGenSamples[ix]:mul(-1)
        end
    end
    local gradQuatAdv = AdvCriterionPose:updateGradInput(poseGenSamples)
    local errPoseAdv = AdvCriterionPose.errG
    for ix=1,params.nCams*params.batchSize do
        if(flipQ[ix]==1) then
            gradQuatAdv[ix]:mul(-1)
        end
    end
    --print(errPoseAdv)
    AdvCriterionPose:updateAdversary(1)
    if(params.quatPriorWt > 0) then
        gradPosePred[2]:add(gradQuatAdv:mul(params.quatPriorWt))
    end
    --print(AdvCriterionPose.realSample:mean(1), AdvCriterionPose.generatedSamples:mean(1))
    
    errPerInst = errPerInst:add(AdvCriterionPose.errG_perInst:mul(params.quatPriorWt))
    local rewFunc = netInit.updateRewardsFunc(errPerInst:clone():div(errPerInst:numel()))
    encoderPose:apply(rewFunc)
    
    encoderPose:backward(imgsObsColor, gradPosePred)
    --print(errPerInst:mean())
    return err, netPoseGradParameters
end
--print(netRecons)
-----------------------------
----------Training-----------
if(params.display) then disp = require 'display' end
for iter=1,params.numTrainIter do
--for iter=1,10 do
    print(iter,err, errQuat, AdvCriterionPose.errG,AdvCriterionPose.errD)
    fout:write(string.format('%d %f\n',iter,err))
    fout:flush()
    if(iter%params.visIter==0) then
        local dispVar = pred[1]:clone()
        if(params.disp == 1) then
            disp.image(imgs, {win=10, title='inputIm'})
            disp.image(dispVar:max(3):squeeze(), {win=1, title='predX',min=0,max=1})
            disp.image(dispVar:max(4):squeeze(), {win=2, title='predY',min=0,max=1})
            disp.image(dispVar:max(5):squeeze(), {win=3, title='predZ',min=0,max=1})
        end
    end
    if(iter%10000)==0 then
        torch.save(params.snapshotDir .. '/netPose'.. iter .. '.t7', encoderPose)
        torch.save(params.snapshotDir .. '/netShape'.. iter .. '.t7', netRecons)
    end
    optim.adam(fx, netParameters, optimState)
    optim.adam(fxPose, netPoseParameters, optimStatePose)
    --print(tot_tm:time().real, data_tm:time().real, loss_tm:time().real) 
end