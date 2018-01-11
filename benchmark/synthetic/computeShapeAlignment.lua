torch.manualSeed(1)
require 'cunn'
require 'optim'
local matio = require 'matio'
local data = dofile('../data/synthetic/shapenet.lua')
local splitUtil = dofile('../benchmark/synthetic/splits.lua')
local nInit = dofile('../nnutils/netInit.lua')
local gridBoundSnet = 0.5 --parameter fixed according to shapenet models' size. Do not change
local gridBoundStn = 1 --trilinear sampler assumes the grid represents a volume in [-1,1]. Do not change.
local extrinsicScale = gridBoundStn/gridBoundSnet
local aligner = dofile('../benchmark/align/shape.lua')

params = {}
params.synset = '3001627' -- car:02958343, chair:03001627, aero:02691156
params.imgSize = torch.Tensor({64, 64})
params.obsSize = torch.Tensor({64, 64})
params.nCams = 2
params.nImgs = 5
params.batchSize = 8
params.netName = 'chair_depth_poseRegUnsup_nds80_np8_euler_prior_nc3'
params.netIter = 50000
params.updateTrans = 1
for k,v in pairs(params) do params[k] = tonumber(os.getenv(k)) or os.getenv(k) or params[k] end

params.synset = '0' .. tostring(params.synset) --to resolve string/number issues in passing bash arguments
params.modelsDataDir = '../cachedir/blenderRenderPreprocess/' .. params.synset .. '/'
params.obsType = 'depth'
params.updateTrans = (params.updateTrans == 1)

local valModels = splitUtil.getSplit(params.synset, nil, params.modelsDataDir)['val']
local dataLoader = data.dataLoader(params.modelsDataDir, params.batchSize, params.nCams, params.nImgs, params.obsSize, params.imgSize, params.obsType, valModels, extrinsicScale)
dataLoader.voxelsDir = '../cachedir/shapenet/modelVoxels/' .. params.synset .. '/'
local imgs, obs, Ks, imgsObsColor, cams, gtVoxels = dataLoader:forward()

local sAlign = aligner.shapeAlign(nn.AbsCriterion(), {32,32,32})
sAlign:cuda()

local netPred = torch.load(paths.concat('../cachedir/snapshots/shapenet', params.netName, 'netShape' .. tostring(params.netIter) .. '.t7'))
netPred:evaluate()
local predsVol = netPred:forward(imgs:cuda())
if(not torch.isTensor(predsVol)) then predsVol = predsVol[1] end

local bestQuat, alignedShapes = sAlign:align(predsVol:cuda(), gtVoxels:cuda(), 50, 600)

local saveDir = paths.concat('../cachedir/alignment/shapenet',params.netName)
paths.mkdir(saveDir)
local saveFile = paths.concat(saveDir, 'shape' .. tostring(params.netIter) .. '.mat' )
if params.updateTrans then
    matio.save(saveFile, {trans=bestQuat[1]:double(), quat=bestQuat[2]:double()})
else
    matio.save(saveFile, {quat=bestQuat[2]:double()})
end
--print(bestQuat)