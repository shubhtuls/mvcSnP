require 'nn'
require 'nngraph'
local M = {}
local quatUtils = dofile('../geometry/quatUtils.lua')
-------------------------------
-------------------------------
-- input is BXnPX3 points, BX4 quauternion dimensions
-- output is BXnPX3 points
local function rotation(nP)
    local points = - nn.Identity()
    local zero = points - nn.Narrow(3,1,1) - nn.MulConstant(0,false)
    local pointsQuat = {zero,points} - nn.JoinTable(3) --prepends zero to 'real' part of points
    
    local quat = - nn.Identity()
    local quatRep = quat - nn.Replicate(nP,2)
    local rot = {pointsQuat, quatRep} - quatUtils.quatRotateModule()
    
    nngraph.annotateNodes()
    local gmod = nn.gModule({points, quat}, {rot})
    return gmod
end
-------------------------------
-------------------------------
-- input is BXnPX3 points, BX3 translation vectors
-- output is BXnPX3 points
local function translation(nP)
    local points = - nn.Identity()
    local trans = - nn.Identity()
    local transRep = trans - nn.Replicate(nP,2)
    
    local final = {points,transRep} - nn.CAddTable()
    
    nngraph.annotateNodes()
    local gmod = nn.gModule({points, trans}, {final})
    return gmod
end
-------------------------------
-------------------------------
-- input is BXnPX3 points, BX3 translation vectors, BX4 quaternions
-- output is BXnPX3 points
-- performs p_out = R*(p_in - t)
local function rigidTransform(nP)
    local points = - nn.Identity()
    local trans = - nn.Identity()
    local quat = - nn.Identity()
    
    local minus_t = trans - nn.MulConstant(-1,false)
    local p1 = {points,minus_t} - translation(nP)
    local p2 = {p1,quat} - rotation(nP)
    
    nngraph.annotateNodes()
    local gmod = nn.gModule({points, trans, quat}, {p2})
    return gmod
end
-------------------------------
-------------------------------
M.rotation = rotation
M.translation = translation
M.rigidTransform = rigidTransform
return M