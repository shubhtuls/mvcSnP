require 'cunn'
require 'optim'
require 'nngraph'
local matio = require 'matio'
local data = dofile('../benchmark/synthetic/data.lua')
local quatUtil = dofile('../geometry/quatUtils.lua')
local netInit = dofile('../nnutils/netInit.lua')
local alignerPose = dofile('../benchmark/align/rotation.lua')
local pAlign = alignerPose.quatAlign()

-----------------------------
--------parameters-----------
local params = {}
--params.bgVal = 0
params.netName = 'name'
params.gpu = 3
params.imgSizeY = 64
params.imgSizeX = 64
params.synset = 2958343 --chair:3001627, aero:2691156, car:2958343
params.nImgs = 2
params.disp = 0
params.evalSet = 'val'
params.netIter = 50000

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(params) do params[k] = tonumber(os.getenv(k)) or os.getenv(k) or params[k] end

if params.disp == 0 then params.disp = false end

params.snapshotDir = '../cachedir/snapshots/shapenet/' .. params.netName
params.imgSize = torch.Tensor({params.imgSizeX, params.imgSizeY})
params.synset = '0' .. tostring(params.synset) --to resolve string/number issues in passing bash arguments
params.modelsDataDir = '../cachedir/blenderRenderPreprocess/' .. params.synset .. '/'
params.voxelsDir = '../cachedir/shapenet/modelVoxels/' .. params.synset .. '/'
params.imgSize = torch.Tensor({params.imgSizeX, params.imgSizeY})
print(params)
-----------------------------
-----------------------------
params.saveDir = '../cachedir/resultsDir/pose/shapenet/' .. params.netName .. '_' .. tostring(params.netIter) .. '_' .. params.evalSet

paths.mkdir(params.saveDir)
cutorch.setDevice(params.gpu)

local splitUtil = dofile('../benchmark/synthetic/splits.lua')
local testModels = splitUtil.getSplit(params.synset)[params.evalSet]
local dataLoader = data.dataLoader(params.modelsDataDir, params.voxelsDir, params.nImgs, params.imgSize, testModels, extrinsicScale)

local predNet = torch.load(params.snapshotDir .. '/netPose'.. params.netIter .. '.t7')
netPose = predNet:cuda()
netPose:evaluate()
netPose:apply(netInit.setTestMode)

local transformQuat = torch.Tensor({1,0,0,0})
local shapeAlignFile = paths.concat('../cachedir/alignment/shapenet',params.netName,'shape' .. tostring(params.netIter) .. '.mat')
if(paths.filep(shapeAlignFile)) then
    transformQuat = matio.load(shapeAlignFile, {'quat'})
    transformQuat = transformQuat.quat:clone():squeeze()
end
transformQuat[1] = -transformQuat[1]

---------------
---------------
local q1s = - nn.Identity()
local q2s = - nn.Identity()
local q1sUnsq = q1s - nn.Unsqueeze(2)
local q2sUnsq = q2s - nn.Unsqueeze(2) - quatUtil.quatConjugateModule()
local qMult = {q1sUnsq,q2sUnsq} - quatUtil.HamiltonProductModule() - nn.Sum(2)
local qMultiplier = nn.gModule({q1s, q2s}, {qMult})
---------------
---------------

--local iterMax = 10
local iterMax = #testModels
local errsAll = torch.Tensor(iterMax,params.nImgs):fill(0)
local predQuatsAll = torch.Tensor(iterMax,params.nImgs,4):fill(0)
local gtQuatsAll = torch.Tensor(iterMax,params.nImgs,4):fill(0)


for modelId =1,iterMax do
    print('modelId : ' .. tostring(modelId))
    local imgs, camData, gtVol = dataLoader:forward()
    local gtQuats = camData[2]
    imgs = imgs:clone():cuda()
    --print(imgs:size())
    local predQuats = netPose:forward(imgs)

    if(not torch.isTensor(predQuats)) then predQuats = predQuats[2] end
    predQuats = predQuats:double()
    
    predQuats = pAlign:transform(predQuats, transformQuat)
    local errQuat = qMultiplier:forward({predQuats, gtQuats})
    
    for b=1,errQuat:size(1) do
        predQuatsAll[modelId][b]:copy(predQuats[b])
        gtQuatsAll[modelId][b]:copy(gtQuats[b])

        if(errQuat[b][1] < 0) then
            errQuat:narrow(1,b,1):mul(-1)
        end
    end
    local errs = torch.acos(errQuat:narrow(2,1,1)):abs():mul(180/math.pi):mul(2)
    --print(predQuats, gtQuats,errQuat,errs)
    errsAll[modelId]:copy(errs)
end

--print(errsAll)
local meanErr = errsAll:mean()
local medErr = torch.median(errsAll:view(-1))
local acc_pi_by_6 = torch.lt(errsAll,30):double():mean()
--matio.save(paths.concat(params.saveDir,'errs.mat'),{errs=errsAll, predQuats=predQuatsAll, gtQuats=gtQuatsAll})
matio.save(paths.concat(params.saveDir,'errs.mat'),{medErr=medErr, acc_pi_by_6=acc_pi_by_6})

print(meanErr, medErr, acc_pi_by_6)