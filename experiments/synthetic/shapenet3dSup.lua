--torch.manualSeed(1)
torch.setnumthreads(1) --code is slower if using more !!
require 'cunn'
require 'optim'
local data = dofile('../data/synthetic/shapenet.lua')
local netBlocks = dofile('../nnutils/netBlocks.lua')
local netInit = dofile('../nnutils/netInit.lua')
local vUtil = dofile('../utils/visUtils.lua')
local splitUtil = dofile('../benchmark/synthetic/splits.lua')
local miscUtil = dofile('../utils/misc.lua')
-----------------------------
--------parameters-----------
local gridBoundSnet = 0.5 --parameter fixed according to shapenet models' size. Do not change
local gridBoundStn = 1 --trilinear sampler assumes the grid represents a volume in [-1,1]. Do not change.
local extrinsicScale = gridBoundStn/gridBoundSnet --do not change
local bgDepth = extrinsicScale*10.0 --parameter fixed according to rendering used. Do not change.

local params = {}
params.name = 'car_3dSup'
params.gpu = 1
params.batchSize = 8
params.nImgs = 5
params.imgSizeY = 64
params.imgSizeX = 64
params.obsSizeY = 32
params.obsSizeX = 32

params.synset = 2958343 --chair:3001627, aero:2691156, car:2958343

params.gridSizeX = 32
params.gridSizeY = 32
params.gridSizeZ = 32

params.imsave = 0
params.disp = 0
params.bottleneckSize = 100
params.visIter = 100
params.nConvEncLayers = 5
params.nConvDecLayers = 4
params.nConvEncChannelsInit = 8
params.numTrainIter = 10000

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
params.modelsDataDir = '../cachedir/blenderRenderPreprocess/' .. params.synset .. '/'
params.nCams = 1
params.obsType = 'mask'

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
local lossFunc = nn.MSECriterion():cuda()
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
----------Recons-------------
local splitUtil = dofile('../benchmark/synthetic/splits.lua')
local trainModels = splitUtil.getSplit(params.synset, nil, params.modelsDataDir)['train']
local dataLoader = data.dataLoader(params.modelsDataDir, params.batchSize, params.nCams, params.nImgs, params.obsSize, params.imgSize, params.obsType, trainModels, extrinsicScale)
dataLoader.voxelsDir = '../cachedir/shapenet/modelVoxels/' .. params.synset .. '/'

local netRecons = nn.Sequential():add(encoder):add(decoder)
netRecons = netRecons:cuda()
local err = 0

-- Optimization parameters
local optimState = {
   learningRate = 0.0001,
   beta1 = 0.9,
}

local netParameters, netGradParameters = netRecons:getParameters()
local imgs, pred, gtVoxels
local loss_tm = torch.Timer()
-----------------------------
-------Training Func---------
-- fX required for training
local fx = function(x)
    netGradParameters:zero()
    imgs, _, _, _, _, gtVoxels = dataLoader:forward()
    imgs = imgs:cuda()
    gtVoxels = gtVoxels:cuda()
    pred = netRecons:forward(imgs)
    err = lossFunc:forward(pred, gtVoxels)
    local gradPred = lossFunc:backward(pred, gtVoxels)
    netRecons:backward(imgs, gradPred)
    return err, netGradParameters
end
--print(netRecons)
-----------------------------
----------Training-----------
if(params.display) then disp = require 'display' end
for iter=1,params.numTrainIter do
--for iter=1,10 do
    print(iter,err)
    fout:write(string.format('%d %f\n',iter,err))
    fout:flush()
    if(iter%params.visIter==0) then
        local dispVar = pred:clone()
        if(params.disp == 1) then
            disp.image(imgs, {win=10, title='inputIm'})
            disp.image(dispVar:max(3):squeeze(), {win=1, title='predX',min=0,max=1})
            disp.image(dispVar:max(4):squeeze(), {win=2, title='predY',min=0,max=1})
            disp.image(dispVar:max(5):squeeze(), {win=3, title='predZ',min=0,max=1})
        end
    end
    if(iter%5000)==0 then
        torch.save(params.snapshotDir .. '/netShape'.. iter .. '.t7', netRecons)
    end
    optim.adam(fx, netParameters, optimState)
    --print(tot_tm:time().real, data_tm:time().real, loss_tm:time().real) 
end