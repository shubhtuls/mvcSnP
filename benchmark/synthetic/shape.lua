require 'cunn'
require 'optim'
require 'nngraph'
local matio = require 'matio'
local data = dofile('../benchmark/synthetic/data.lua')
local quatUtil = dofile('../geometry/quatUtils.lua')
local netInit = dofile('../nnutils/netInit.lua')
local alignerShape = dofile('../benchmark/align/shape.lua')
local sAlign = alignerShape.shapeAlign(nn.AbsCriterion(), {32,32,32})

-----------------------------
--------parameters-----------
local params = {}
--params.bgVal = 0
params.netName = 'name'
params.gpu = 3
params.imgSizeY = 64
params.imgSizeX = 64
params.synset = 2958343 --chair:3001627, aero:2691156, car:2958343
params.nImgsTest = 2
params.disp = 0
params.evalSet = 'val'
params.netIter = 50000

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(params) do params[k] = tonumber(os.getenv(k)) or os.getenv(k) or params[k] end

if params.disp == 0 then params.disp = false end

params.nImgs = 8
params.snapshotDir = '../cachedir/snapshots/shapenet/' .. params.netName
params.imgSize = torch.Tensor({params.imgSizeX, params.imgSizeY})
params.synset = '0' .. tostring(params.synset) --to resolve string/number issues in passing bash arguments
params.modelsDataDir = '../cachedir/blenderRenderPreprocess/' .. params.synset .. '/'
params.voxelsDir = '../cachedir/shapenet/modelVoxels/' .. params.synset .. '/'
params.imgSize = torch.Tensor({params.imgSizeX, params.imgSizeY})
print(params)
-----------------------------
-----------------------------
params.saveDir = '../cachedir/resultsDir/shape/shapenet/' .. params.netName .. '_' .. tostring(params.netIter) .. '_' .. params.evalSet

paths.mkdir(params.saveDir)
cutorch.setDevice(params.gpu)

local splitUtil = dofile('../benchmark/synthetic/splits.lua')
local testModels = splitUtil.getSplit(params.synset)[params.evalSet]
local dataLoader = data.dataLoader(params.modelsDataDir, params.voxelsDir, params.nImgs, params.imgSize, testModels, extrinsicScale)

local predNet = torch.load(params.snapshotDir .. '/netShape'.. params.netIter .. '.t7')
netShape = predNet:cuda()
netShape:evaluate()
netShape:apply(netInit.setTestMode)

local transformQuat = torch.Tensor({1,0,0,0})
local transformTrans = torch.Tensor({0,0,0})
local shapeAlignFile = paths.concat('../cachedir/alignment/shapenet',params.netName,'shape' .. tostring(params.netIter) .. '.mat')
if(paths.filep(shapeAlignFile)) then
    local transformVars = matio.load(shapeAlignFile)
    transformQuat = transformVars.quat:clone():squeeze()
    if (transformVars.trans) then
        transformTrans = transformVars.trans:clone():squeeze()
    end
end
transformQuat = transformQuat:cuda()
transformTrans = transformTrans:cuda()
sAlign:cuda()

local counter = 1
--local iterMax = 10
local iterMax = #testModels
for modelId =1,iterMax do
    print('modelId : ' .. tostring(modelId))
    local imgs, camData, gtVol = dataLoader:forward()
    imgs = imgs:clone():cuda()
    local predVol = netShape:forward(imgs)
    if(not torch.isTensor(predVol)) then predVol = predVol[1] end
    
    local predVolOrig = predVol:clone():double()
    predVol = sAlign:transform(predVol, {transformTrans, transformQuat}):double()
    
    for ix = 1, params.nImgsTest do
        matio.save(paths.concat(params.saveDir, tostring(counter) .. '.mat'),{gtName=testModels[modelId], gtVol=gtVol[ix][1],volume=predVol[ix][1],volumeUnaligned=predVolOrig[ix][1],img=dataLoader.imgsOriginal[ix]:clone():double()})
        counter = counter + 1
    end
end