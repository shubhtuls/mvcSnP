--torch.manualSeed(1)
torch.setnumthreads(1) --code is slower if using more !!
require 'cunn'
require 'optim'
local data = dofile('../data/synthetic/shapenet.lua')
local netBlocks = dofile('../nnutils/netBlocks.lua')
local netInit = dofile('../nnutils/netInit.lua')
local splitUtil = dofile('../benchmark/synthetic/splits.lua')
local miscUtil = dofile('../utils/misc.lua')
local quatUtil = dofile('../geometry/quatUtils.lua')
dofile('../loss/quatLoss.lua')
-----------------------------
--------parameters-----------
local gridBoundSnet = 0.5 --parameter fixed according to shapenet models' size. Do not change
local gridBoundStn = 1 --trilinear sampler assumes the grid represents a volume in [-1,1]. Do not change.
local extrinsicScale = gridBoundStn/gridBoundSnet --do not change
local bgDepth = extrinsicScale*10.0 --parameter fixed according to rendering used. Do not change.
local useCudaLoss = true

local params = {}
--params.bgVal = 0
params.name = 'car_posePred'
params.gpu = 1
params.batchSize = 8
params.nImgs = 5
params.nCams = 5
params.imgSizeY = 64
params.imgSizeX = 64
params.obsSizeY = 32
params.obsSizeX = 32
params.nPoses = 1
params.maxTrainModels = 0

params.bgWt = 0.2 -- figured out via cross-validation on the val set. Code currently ignoring this.
params.synset = 2958343 --chair:3001627, aero:2691156, car:2958343

params.gridSizeX = 32
params.gridSizeY = 32
params.gridSizeZ = 32

params.imsave = 0
params.disp = 0
params.obsType = 'mask'
params.bottleneckSize = 100
params.visIter = 100
params.nConvEncLayers = 5
params.nConvDecLayers = 4
params.nConvEncChannelsInit = 8
params.numTrainIter = 50000
params.quatSupWt = 1

params.useEuler = 1
params.elPredMin = -20
params.elPredRange = 60
params.elPriorMin = -20
params.elPriorRange = 60

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(params) do params[k] = tonumber(os.getenv(k)) or os.getenv(k) or params[k] end

if params.disp == 0 then params.display = false else params.display = true end
if params.imsave == 0 then params.imsave = false end
if params.maxTrainModels == 0 then params.maxTrainModels = nil end
params.visDir = '../cachedir/visualization/shapenet/' .. params.name
params.snapshotDir = '../cachedir/snapshots/shapenet/' .. params.name
params.imgSize = torch.Tensor({params.imgSizeY, params.imgSizeX})
params.obsSize = torch.Tensor({params.obsSizeY, params.obsSizeX})
params.gridSize = torch.Tensor({params.gridSizeX, params.gridSizeY, params.gridSizeZ})
params.synset = '0' .. tostring(params.synset) --to resolve string/number issues in passing bash arguments
params.modelsDataDir = '../cachedir/blenderRenderPreprocess/' .. params.synset .. '/'
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
local lossFuncQuat = nn.QuatCriterion()
lossFuncQuat = lossFuncQuat:cuda()

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
encoderPose:add(bottleneckPose):add(nn.Reshape(params.bottleneckSize,true)):add(quatUtil.quatPredSampleModule(params.bottleneckSize,params.nPoses,params.useEuler == 1,params.elPredMin,params.elPredRange))
encoderPose:apply(netInit.weightsInit)
encoderPose.modules[#encoderPose.modules]:apply(netInit.weightsInitPose)
print(encoderPose)
--print(encoder)
-----------------------------
----------Recons-------------
local splitUtil = dofile('../benchmark/synthetic/splits.lua')
local trainModels = splitUtil.getSplit(params.synset, nil, params.modelsDataDir)['train']
local dataLoader = data.dataLoader(params.modelsDataDir, params.batchSize, params.nCams, params.nImgs, params.obsSize, params.imgSize, params.obsType, trainModels, extrinsicScale)
encoderPose = encoderPose:cuda()
local err, errQuat = 0, 0

local optimStatePose = {
   learningRate = 0.0001,
   beta1 = 0.9,
}

local netPoseParameters, netPoseGradParameters = encoderPose:getParameters()
local imgs, pred, observations, camKs, imgsObsColor, camData, gradPosePred, posePred, errPerInst

-----------------------------
-------Training Func---------
local fxPose = function(x)
    netPoseGradParameters:zero()
    imgs, observations, camKs, imgsObsColor, camData = dataLoader:forward(params.maxTrainModels)
    imgsObsColor = imgsObsColor:reshape(params.nCams*params.batchSize,3,imgsObsColor:size(4),imgsObsColor:size(5)):cuda()
    posePred = encoderPose:forward(imgsObsColor)
    gradPosePred = posePred:clone():fill(0)
    errPerInst = torch.Tensor(params.nCams*params.batchSize)

    local poseGt = camData[2]:clone():reshape(params.nCams*params.batchSize,4):cuda()
    errQuat = lossFuncQuat:forward(posePred:cuda(), poseGt)
    local gradQuat = lossFuncQuat:backward(posePred:cuda(), poseGt)
    gradPosePred:add(gradQuat:mul(params.quatSupWt))
    
    local errPerInst = lossFuncQuat.errs:clone()
    local rewFunc = netInit.updateRewardsFunc(errPerInst:clone():div(errPerInst:numel()))
    encoderPose:apply(rewFunc)

    encoderPose:backward(imgsObsColor, gradPosePred)
    return err, netPoseGradParameters
end
--print(netRecons)
-----------------------------
----------Training-----------
if(params.display) then disp = require 'display' end
for iter=1,params.numTrainIter do
--for iter=1,10 do
    print(iter, errQuat)
    fout:write(string.format('%d %f\n',iter,errQuat))
    fout:flush()
    if(iter%5000)==0 then
        torch.save(params.snapshotDir .. '/netPose'.. iter .. '.t7', encoderPose)
    end
    optim.adam(fxPose, netPoseParameters, optimStatePose)
    --print(tot_tm:time().real, data_tm:time().real, loss_tm:time().real) 
end