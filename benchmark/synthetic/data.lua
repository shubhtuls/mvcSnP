local M = {}
require 'image'
local cropUtils = dofile('../utils/cropUtils.lua')
local matio = require 'matio'
local quatUtils = dofile('../geometry/quatUtils.lua')
-------------------------------
-------------------------------
local maxSavedDepth = 10
local function BuildArray(...)
  local arr = {}
  for v in ... do
    arr[#arr + 1] = v
  end
  return arr
end
-------------------------------
-------------------------------
local dataLoader = {}
dataLoader.__index = dataLoader

setmetatable(dataLoader, {
    __call = function (cls, ...)
        return cls.new(...)
    end,
})

function dataLoader.new(synsetDir, voxelsDir, nImgs, imgSize, modelNames, extrinsicScale)
    local self = setmetatable({}, dataLoader)
    self.bs = nImgs
    self.voxelsDir = voxelsDir
    self.imgSize = imgSize
    self.synsetDir = synsetDir
    self.modelNames = modelNames
    self.extrinsicScale = extrinsicScale
    self.mId = 1
    
    self.imgsOriginal = torch.Tensor(self.bs,3,224,224)
    return self
end

function dataLoader:forward()
    
    local imgs = torch.Tensor(self.bs, 3, self.imgSize[1], self.imgSize[2]):fill(0)
    local gtVoxels = torch.Tensor(self.bs, 1, 32, 32, 32):fill(0)
    
    local extrinsicMats = torch.Tensor(self.bs, 4, 4)
    local quats = torch.Tensor(self.bs, 4)
    local translations = torch.Tensor(self.bs, 3)
    
    for b = 1,self.bs do
        local mId = self.mId
        local imgsDir = paths.concat(self.synsetDir, self.modelNames[mId])
        local inpImgNum = b-1
        local imgRgb = image.load(string.format('%s/render_%d.png',imgsDir,inpImgNum))
        
        local alphaMask = imgRgb[4]:repeatTensor(3,1,1)
        imgRgb = torch.cmul(imgRgb:narrow(1,1,3),alphaMask) + 1 - alphaMask
        self.imgsOriginal[b]:copy(imgRgb)
        
        imgRgb = image.scale(imgRgb,self.imgSize[2], self.imgSize[1])
        imgs[b]:copy(imgRgb)
        
        local voxelFile = paths.concat(self.voxelsDir, self.modelNames[mId] .. '.mat')
        gtVoxels[b][1]:copy(matio.load(voxelFile,{'Volume'})['Volume']:typeAs(gtVoxels))
        
        local imgNum = b - 1
        -- read camera data. Also output extrinsic pose if available so we can debug
        local camData = matio.load(string.format('%s/camera_%d.mat',imgsDir,imgNum),{'K','extrinsic'})
            
        extrinsicMats[b]:copy(camData.extrinsic)
        quats[b]:copy(quatUtils.rot2quat(camData.extrinsic:narrow(1,1,3):narrow(2,1,3):transpose(2,1)))
        translations[b]:copy(camData.extrinsic:narrow(1,1,3):narrow(2,4,1))
    end

    -- the data data coordinate frame scale is different from the prediction coordinate
    if(self.extrinsicScale) then
        translations:mul(self.extrinsicScale)
        extrinsicMats:narrow(2,1,3):narrow(3,4,1):mul(self.extrinsicScale)
    end
    self.mId = self.mId+1
    return imgs, {extrinsicMats, quats, translations}, gtVoxels:transpose(5,4)
end
-------------------------------
-------------------------------
M.dataLoader = dataLoader
return M