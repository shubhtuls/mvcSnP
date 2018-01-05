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

function dataLoader.new(synsetDir, bs, nCams, nImgs, obsSize, imgSize, obsType, modelNames, extrinsicScale)
    local self = setmetatable({}, dataLoader)
    self.bs = bs
    self.nCams = nCams --number of observations per iteration per instance
    self.nImgs = nImgs --number of renderings to sample inputs/observations from
    self.obsSize = obsSize
    self.imgSize = imgSize
    self.synsetDir = synsetDir
    self.obsType = obsType --'color', or 'mask', or 'depth', or 'cpm' (color plus mask)
    self.modelNames = modelNames
    self.extrinsicScale = extrinsicScale
    return self
end

function dataLoader:forward(maxModelInd)
    
    local maxModelInd = maxModelInd or #self.modelNames
    local imgs = torch.Tensor(self.bs, 3, self.imgSize[1], self.imgSize[2]):fill(0)
    local imgsObsColor = torch.Tensor(self.nCams, self.bs, 3, self.imgSize[1], self.imgSize[2]):fill(0)
    local gtVoxels = torch.Tensor(self.bs, 1, 32, 32, 32):fill(0) --debugging purposes
    
    local ncObs = 1
    local observations = torch.Tensor(self.nCams, self.bs, ncObs, self.obsSize[1], self.obsSize[2])

    local kMats = torch.Tensor(self.nCams, self.bs, 3, 3)
    
    local extrinsicMats = torch.Tensor(self.nCams, self.bs, 4, 4)
    local quats = torch.Tensor(self.nCams, self.bs, 4)
    local translations = torch.Tensor(self.nCams, self.bs, 3)
    
    for b = 1,self.bs do
        local mId = torch.random(1,maxModelInd)
        local imgsDir = paths.concat(self.synsetDir, self.modelNames[mId])
        local nImgs = self.nImgs
        local inpImgNum = torch.random(0,nImgs-1)
        local imgRgb = image.load(string.format('%s/render_%d.png',imgsDir,inpImgNum))
        
        imgRgb = image.scale(imgRgb,self.imgSize[2], self.imgSize[1])
        local alphaMask = imgRgb[4]:repeatTensor(3,1,1)
        imgRgb = torch.cmul(imgRgb:narrow(1,1,3),alphaMask) + 1 - alphaMask

        imgs[b] = imgRgb
        
        if(self.voxelsDir) then
            local voxelFile = paths.concat(self.voxelsDir, self.modelNames[mId] .. '.mat')
            gtVoxels[b][1] = matio.load(voxelFile,{'Volume'})['Volume']:typeAs(gtVoxels)
        end
        
        local rPerm = torch.randperm(nImgs)
        for nc = 1,self.nCams do
            local imgNum = rPerm[nc] - 1
            local obsSizeOrig --used so we can correct the K matrix to account for resizing
            -- read observation images
            if(self.obsType == 'depth') then
                local depthObs = image.load(string.format('%s/%s_%d.png',imgsDir,'depth',imgNum))*maxSavedDepth
                obsSizeOrig = depthObs:size()
                depthObs = image.scale(depthObs,self.obsSize[2], self.obsSize[1],'simple')
                observations[nc][b]:copy(depthObs)
            elseif(self.obsType == 'mask') then
                local depthObs = image.load(string.format('%s/%s_%d.png',imgsDir,'depth',imgNum))*maxSavedDepth
                obsSizeOrig = depthObs:size()
                depthObs = image.scale(depthObs,self.obsSize[2], self.obsSize[1])
                local maskObs = torch.ge(depthObs, maxSavedDepth - 1e-6):double()
                observations[nc][b]:narrow(1,ncObs,1):copy(maskObs)
            end
            
            -- Load color image corresponding to the observation viewpoint, we'll use this to predict pose
            local colorIm = image.load(string.format('%s/%s_%d.png',imgsDir,'render',imgNum))
            local alphaMask = colorIm[4]:repeatTensor(3,1,1)
            
            colorIm = torch.cmul(colorIm:narrow(1,1,3),alphaMask) + 1 - alphaMask
            local colorImObs = colorIm:clone()
            imgsObsColor[nc][b]:copy(image.scale(colorImObs,self.imgSize[2], self.imgSize[1]))
            
            -- read camera data. Also output extrinsic pose if available so we can debug
            local camData = matio.load(string.format('%s/camera_%d.mat',imgsDir,imgNum),{'K','extrinsic'})
            
            local kMat = camData.K:clone()
            kMat:narrow(1,1,1):mul(self.obsSize[2]/obsSizeOrig[3])
            kMat:narrow(1,2,1):mul(self.obsSize[1]/obsSizeOrig[2])
            kMats[nc][b]:copy(kMat)
            
            extrinsicMats[nc][b]:copy(camData.extrinsic)
            quats[nc][b]:copy(quatUtils.rot2quat(camData.extrinsic:narrow(1,1,3):narrow(2,1,3):transpose(2,1)))
            translations[nc][b]:copy(camData.extrinsic:narrow(1,1,3):narrow(2,4,1))
        end
    end

    -- the data data coordinate frame scale is different from the prediction coordinate
    if(self.extrinsicScale) then
        --print(self.extrinsicScale)
        translations:mul(self.extrinsicScale)
        extrinsicMats:narrow(3,1,3):narrow(4,4,1):mul(self.extrinsicScale)
        if(self.obsType == 'depth') then observations:mul(self.extrinsicScale) end
    end
    return imgs, observations, kMats, imgsObsColor, {extrinsicMats, quats, translations}, gtVoxels:transpose(5,4)
end
-------------------------------
-------------------------------
M.dataLoader = dataLoader
return M