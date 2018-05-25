require 'sys'
require 'image'
require 'os'
local M = {}
-----------------------
-----------------------
local cubeV = torch.Tensor({{0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 1.0, 0.0}, {0.0, 1.0, 1.0}, {1.0, 0.0, 0.0}, {1.0, 0.0, 1.0}, {1.0, 1.0, 0.0}, {1.0, 1.0, 1.0}})
local cubeV = cubeV - 0.5

local cubeF = torch.Tensor({{1,  7,  5 }, {1,  3,  7 }, {1,  4,  3 }, {1,  2,  4 }, {3,  8,  7 }, {3,  4,  8 }, {5,  7,  8 }, {5,  8,  6 }, {1,  5,  6 }, {1,  6,  2 }, {2,  6,  8 }, {2,  8,  4}})

local cubeE = torch.Tensor({{1, 2}, {1, 3}, {1, 5}, {2, 4}, {2, 6}, {3, 4}, {3, 7}, {4, 8}, {5, 6}, {5, 7}, {6, 8}, {7, 8}})

local function voxelsToMesh(predVol, thresh)
    local vCounter = 1
    local fCounter = 1
    local totPoints = torch.gt(predVol, thresh):sum()
    
    local vAll = cubeV:repeatTensor(totPoints, 1)
    local probAll = torch.Tensor(totPoints*12):fill(0)
    local fAll = cubeF:repeatTensor(totPoints, 1)
    
    local fOffset = torch.repeatTensor(torch.linspace(0,12*totPoints-1,12*totPoints),3,1):transpose(1,2)
    fOffset = torch.floor(fOffset/12)*8
    fAll = fAll + fOffset
    
    for x=1,predVol:size(1) do
        for y=1,predVol:size(2) do
            for z=1,predVol:size(3) do
                --print(x,y,z)
                if predVol[x][y][z] > thresh then
                    --local radius = predVol[x][y][z]
                    local radius = 1
                    probAll:narrow(1,fCounter,12):fill(predVol[x][y][z])
                    vAll:narrow(1,vCounter,8):narrow(2,1,1):mul(radius):add(x)
                    vAll:narrow(1,vCounter,8):narrow(2,2,1):mul(radius):add(y)
                    vAll:narrow(1,vCounter,8):narrow(2,3,1):mul(radius):add(z)
                    vCounter = vCounter+8
                    fCounter = fCounter+12
                end
            end
        end
    end
    return vAll, fAll, probAll
end
-----------------------
-----------------------
local function appendObjAlphas(meshFileHandle, vertices, faces, alphas)
    for vx = 1,vertices:size(1) do
        meshFileHandle:write(string.format('v %f %f %f\n', vertices[vx][1], vertices[vx][2], vertices[vx][3]))
    end
    for fx = 1,faces:size(1) do
        meshFileHandle:write(string.format('usemtl a%.2f\n', alphas[fx]))
        meshFileHandle:write(string.format('f %d %d %d\n', faces[fx][1], faces[fx][2], faces[fx][3]))
    end
end

local function appendObj(meshFileHandle, vertices, faces)
    for vx = 1,vertices:size(1) do
        meshFileHandle:write(string.format('v %f %f %f\n', vertices[vx][1], vertices[vx][2], vertices[vx][3]))
    end
    for fx = 1,faces:size(1) do
        meshFileHandle:write(string.format('f %d %d %d\n', faces[fx][1], faces[fx][2], faces[fx][3]))
    end
end

local function appendObjTex(meshFileHandle, vertices, faces, verticesTex, facesTex)
    for vx = 1,vertices:size(1) do
        meshFileHandle:write(string.format('v %f %f %f\n', vertices[vx][1], vertices[vx][2], vertices[vx][3]))
    end
    for vx = 1,verticesTex:size(1) do
        meshFileHandle:write(string.format('vt %f %f\n', verticesTex[vx][1], verticesTex[vx][2]))
    end
    for fx = 1,faces:size(1) do
        meshFileHandle:write(string.format('f %d/%d %d/%d %d/%d\n', faces[fx][1], facesTex[fx][1], faces[fx][2], facesTex[fx][2], faces[fx][3], facesTex[fx][3]))
    end
end

local function appendAlphaMtl(idList, colors, mtlFileHandle, illumId, alphas)
    local illumId = illumId or 2
    for ax = 1, idList:size(1) do
        local id = idList[ax]
        local color = (colors:numel()==3) and colors or colors[ax]
        local alpha = alphas and alphas[ax] or idList[ax]
        
        mtlFileHandle:write(string.format('newmtl a%.2f\n', id))
        mtlFileHandle:write(string.format('Ka %f %f %f\n', color[1], color[2], color[3]))
        mtlFileHandle:write(string.format('Kd %f %f %f\n', color[1], color[2], color[3]))
        mtlFileHandle:write(string.format('Ks 1 1 1\n'))
        mtlFileHandle:write(string.format('d %f\n', alpha))
        mtlFileHandle:write(string.format('illum %d\n',illumId))
    end
end
-----------------------
-----------------------
local function lineMesh(src, dst, thickness)
    local thickness = thickness or 0.01
    local Vs = cubeV:clone():mul(thickness)
    local Fs = cubeF:clone()
    for d=1,3 do
        Vs:narrow(1,1,4):narrow(2,d,1):add(src[d])
        Vs:narrow(1,5,4):narrow(2,d,1):add(dst[d])
    end
    return Vs, Fs
end

-- (0,0), (1,0), (1,1), (0,1)
local function planeTextureMesh()
    local Fs = torch.Tensor({{1,2,3}, {1,4,3}})
    local VsTex = torch.Tensor({{0,0}, {1,0}, {1,1}, {0,1}})
    return VsTex, Fs
end
-----------------------
-----------------------
local function writeGridMesh(meshFile)
    local gridAlphas = torch.Tensor(12):fill(0.1)
    local gridColor = torch.Tensor({0.6,0.6,0.6})
    local alphaList = torch.linspace(0,1,101)

    local mtlFile = meshFile:split('.obj')[1] .. '.mtl'
    local fout = io.open(meshFile, 'w')
    local foutMtl = io.open(mtlFile, 'w')

    mtlFile = mtlFile:split('/')
    mtlFile = mtlFile[#mtlFile]
    appendAlphaMtl(alphaList, gridColor, foutMtl, 1)
    foutMtl:close()
    fout:write(string.format('mtllib %s\n',mtlFile))
    appendObjAlphas(fout, cubeV, cubeF, gridAlphas)

    local gridBoxLines = true
    fout:write(string.format('usemtl a%.2f\n',0.5))
    local gridThickness = 5e-3
    if(gridBoxLines) then
        local vCounter = 8
        for e=1,12 do
            local vAll, fAll = lineMesh(cubeV[cubeE[e][1]], cubeV[cubeE[e][2]], gridThickness)
            appendObj(fout, vAll, fAll:add(vCounter))
            vCounter = vCounter+8
        end 
    end
    
    fout:close()
end
-----------------------
-----------------------
local function writeTextureMesh(meshFile, texImg, K, extMat, scale)
    local imgSize = texImg:size()
    local bdryPix = torch.Tensor({{imgSize[2],imgSize[3],1}, {0, imgSize[3], 1}, {0, 0, 1}, {imgSize[2],0,1}})
    local scale = scale or 1.
    local Kinv = torch.inverse(K)
    local Einv = torch.inverse(extMat)
    local bdryDir = torch.mm(bdryPix, Kinv:transpose(1,2))
        
    local Cframe = torch.cat(bdryDir:clone():mul(scale), torch.Tensor(4,1):fill(1))
    local vAll = torch.mm(Cframe, Einv:transpose(1,2)):narrow(2,1,3)

    local mtlFile = meshFile:split('.obj')[1] .. '.mtl'
    local texFile = meshFile:split('.obj')[1] .. '_texture.png'
    local texFileSuffix = texFile:split('/')
    texFileSuffix = texFileSuffix[#texFileSuffix]
    
    local VsTex, Fs = planeTextureMesh()

    local fout = io.open(meshFile, 'w')
    local foutMtl = io.open(mtlFile, 'w')

    foutMtl:write('newmtl m0\n Ka 0 0 0\n Kd 0 0 0 \n Ks 0 0 0\n')
    foutMtl:write('newmtl mTex\n Ka 1 1 1\n Kd 1 1 1 \n Ks 0 0 0\n')
    foutMtl:write('map_Disp ' .. texFileSuffix .. '\n')
    foutMtl:write('map_Kd ' .. texFileSuffix .. '\n')

    mtlFile = mtlFile:split('/')
    mtlFile = mtlFile[#mtlFile]
    fout:write(string.format('mtllib %s\n usemtl mTex\n',mtlFile))
    vAll = torch.mm(vAll, torch.Tensor({{1,0,0}, {0,0,1}, {0,1,0}}))

    appendObjTex(fout, vAll, Fs, VsTex, Fs)

    fout:write(string.format('mtllib %s\n usemtl m0\n',mtlFile))
    local vCounter = 4
    local srcs = {vAll[1], vAll[1],vAll[3], vAll[3]}
    local dsts = {vAll[2], vAll[4],vAll[2], vAll[4]}
    for sx = 1,#srcs do
        vAllLine, fAllLine = lineMesh(srcs[sx], dsts[sx])
        appendObj(fout, vAllLine, fAllLine:add(vCounter))
        vCounter = vCounter+8
    end        

    image.save(texFile, image.hflip(texImg))
    --image.save(texFile, imgs[b])

    foutMtl:close()
    fout:close()
end
-----------------------
-----------------------
function renderMesh(meshFile, pngFile, az, el, distScale,upsamp,theta)
    local az = az or 60
    local el = el or 20
    local distScale = distScale or 1.0
    local upsamp = upsamp or 1.0
    local theta = theta or 0
    local blendFile = '/data0/shubhtuls/code/mvcSnP/renderer/model.blend'
    local blenderExec = '/home/shubhtuls/Downloads/blender-2.79a/blender'
    local command = string.format('bash ../renderer/render.sh %s %s %s %s %d %d %f %f %d',blenderExec, blendFile, meshFile, pngFile, az, el, distScale, upsamp, theta)
    --print(command)
    os.execute(command)
    local img = image.load(pngFile)
    local alphaMask = img[4]:repeatTensor(3,1,1)
    img = torch.cmul(img:narrow(1,1,3),alphaMask) + 1 - alphaMask
    return img
end
-----------------------
-----------------------
local function voxelsToColorMesh(predVol, predColors, thresh)
    local vCounter = 1
    local fCounter = 1
    local totPoints = torch.gt(predVol, thresh):sum()
    
    local vAll = cubeV:repeatTensor(totPoints, 1)
    local fAll = cubeF:repeatTensor(totPoints, 1)
    local colorsAll = torch.Tensor(fAll:size())
    local colorsVert = torch.Tensor(vAll:size())
    
    print(colorsAll:size())
    
    local fOffset = torch.repeatTensor(torch.linspace(0,12*totPoints-1,12*totPoints),3,1):transpose(1,2)
    fOffset = torch.floor(fOffset/12)*8
    fAll = fAll + fOffset
    
    for x=1,predVol:size(1) do
        for y=1,predVol:size(2) do
            for z=1,predVol:size(3) do
                --print(x,y,z)
                if predVol[x][y][z] > thresh then
                    local radius = predVol[x][y][z]
                    vAll:narrow(1,vCounter,8):narrow(2,1,1):mul(radius):add(x)
                    vAll:narrow(1,vCounter,8):narrow(2,2,1):mul(radius):add(y)
                    vAll:narrow(1,vCounter,8):narrow(2,3,1):mul(radius):add(z)
                    local colorPt = predColors[{{},x,y,z}]
                    local colorRep = torch.repeatTensor(colorPt:squeeze(),12,1)
                    colorsAll:narrow(1,fCounter,12):copy(colorRep)
                    
                    colorsVert:narrow(1,vCounter,8):copy(colorRep:narrow(1,1,8))
                    
                    vCounter = vCounter+8
                    fCounter = fCounter+12
                end
            end
        end
    end
    return vAll, fAll, colorsAll, colorsVert
end

-----------------------
-----------------------
M.appendObjAlphas = appendObjAlphas
M.appendAlphaMtl = appendAlphaMtl
M.voxelsToMesh = voxelsToMesh
M.cubeV = cubeV
M.cubeF = cubeF
M.cubeE = cubeE
M.appendObj = appendObj
M.appendObjTex = appendObjTex
M.lineMesh = lineMesh
M.planeTextureMesh = planeTextureMesh
M.writeGridMesh  = writeGridMesh
M.writeTextureMesh = writeTextureMesh
M.renderMesh = renderMesh
M.voxelsToColorMesh = voxelsToColorMesh
return M