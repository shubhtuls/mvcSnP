local M = {}
----------

local function arrayMontage(images)
  local nperrow = math.ceil(math.sqrt(#images))

  local maxsize = {1, 0, 0}
  for i, img in ipairs(images) do
    if img:dim() == 2 then
      img = torch.expand(img:view(1, img:size(1), img:size(2)), maxsize[1], img:size(1), img:size(2))
    end
    images[i] = img
    maxsize[1] = math.max(maxsize[1], img:size(1))
    maxsize[2] = math.max(maxsize[2], img:size(2))
    maxsize[3] = math.max(maxsize[3], img:size(3))
  end

  -- merge all images onto one big canvas
  local numrows = math.ceil(#images / nperrow)
  local canvas = torch.FloatTensor(maxsize[1], maxsize[2] * numrows, maxsize[3] * nperrow):fill(0.5)
  local row = 0
  local col = 0
  for i, img in ipairs(images) do
    canvas:narrow(2, maxsize[2] * row + 1, img:size(2)):narrow(3, maxsize[3] * col + 1, img:size(3)):copy(img)
    col = col + 1
    if col == nperrow then
      col = 0
      row = row + 1
    end
  end
  return canvas
end

local function montage(img)
  if type(img) == 'table' then
    return arrayMontage(img)
  end

  -- img is a collection?
  if img:dim() == 4 or (img:dim() == 3 and img:size(1) > 3) then
    local images = {}
    for i = 1,img:size(1) do
      images[i] = img[i]
    end
    return arrayMontage(images)
  end
  return img
end

local function imsave(img,path)
    require 'image'
    local saveIm = montage(img)
    -- print(saveIm:size())
    image.save(path,saveIm)
end
----------
local function denseCellSample3D(grid, inds, nSamples)
    local indSamples = {}
    if(nSamples == 1) then 
        return grid:gridIndToPoint(inds + 0.5):reshape(3,1)
    end
    for d=1,3 do
        indSamples[d] = torch.linspace(inds[d],inds[d]+1,nSamples)
    end
    local nS = nSamples^3
    local np = 0
    local points = torch.Tensor(nS,3)
    for ix = 1,nSamples do
        local x = indSamples[1][ix]
        for iy = 1,nSamples do
            local y = indSamples[2][iy]
            for iz = 1,nSamples do
                local z = indSamples[3][iz]
                np = np+1
                points[np] = grid:gridIndToPoint(torch.Tensor({x,y,z}))
            end
        end
    end
    return points:transpose(2,1)
end

local function predGridPoints3D(pred, grid, thresh, nSamples)
    local points = nil
    for ix = 1,pred:size(1) do
        for iy = 1,pred:size(2) do
            for iz = 1,pred:size(3) do
                if(pred[ix][iy][iz] > thresh) then
                    --print(ix,iy,iz)
                    local samples = denseCellSample3D(grid, torch.Tensor({ix,iy,iz}), nSamples)
                    if (points == nil) then
                        points = samples
                    else
                        points = torch.cat(points, samples)
                    end
                end
            end
        end
    end
    --print(pred:sum())
    --print(points:size())
    return points
end
----------
local function InstanceDepthMLE(pred)
    local depth = torch.Tensor(pred:size(1),pred:size(2)):fill(0)
    for ix = 1,pred:size(1) do
        for iy = 1,pred:size(2) do
            local probMax = 0
            local probMult = 1
            local depthInd = 0
            for iz = 1,pred:size(3) do
                local probHit = probMult*(1-pred[ix][iy][iz])
                probMult = probMult*pred[ix][iy][iz]
                if(probHit > probMax) then
                    --print(probHit,probMax,iz)
                    probMax = probHit
                    depthInd = iz
                end
            end
            if(probMax < probMult) then depthInd = 0 end
            depth[ix][iy] = depthInd
        end
    end
    return depth
end

local function depthMLE(pred)
    if(pred:dim()==3) then return InstanceDepthMLE(pred) end
    assert(pred:dim()==4)
    local depth = torch.Tensor(pred:size(1),pred:size(2),pred:size(3)):fill(0)
    for b=1,pred:size(1) do
        depth[b]:copy(InstanceDepthMLE(pred[b]))
    end
    return depth
end
----------
M.imsave = imsave
M.depthMLE = depthMLE
M.predGridPoints3D = predGridPoints3D
return M