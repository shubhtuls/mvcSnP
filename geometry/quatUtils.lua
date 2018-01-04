require 'nn'
require 'nngraph'
if not (nn.ReinforceCategorical) then
    dofile('../nnutils/ReinforceCategorical.lua')
end
if not (nn.TrigCosine) then
    dofile('../geometry/trig.lua')
end
local M = {}
-------------------------------
-------------------------------
-- input is {NXPX4, NXPX4}. output is {NXPX4}
local function HamiltonProductModule()
    local q1 = -nn.Identity()
    local q2 = -nn.Identity()
    
    local inds = torch.Tensor({
            1,-2,-3,-4,
            2,1,4,-3,
            3,-4,1,2,
            4,3,-2,1
        }):reshape(4,4)
    local sign = inds:clone():sign()
    inds = inds:clone():abs()
    
    local q1_q2_prods = {}
    
    for d=1,4 do
        local q2_v1 = q2 - nn.Narrow(3,inds[d][1],1) - nn.MulConstant(sign[d][1], false)
        local q2_v2 = q2 - nn.Narrow(3,inds[d][2],1) - nn.MulConstant(sign[d][2], false)
        local q2_v3 = q2 - nn.Narrow(3,inds[d][3],1) - nn.MulConstant(sign[d][3], false)
        local q2_v4 = q2 - nn.Narrow(3,inds[d][4],1) - nn.MulConstant(sign[d][4], false)
        local q2Sel = {q2_v1, q2_v2, q2_v3, q2_v4} - nn.JoinTable(3)
        q1_q2_prods[d] = {q1, q2Sel} - nn.CMulTable() - nn.Sum(3) - nn.Unsqueeze(3)
    end
    
    local qMult = q1_q2_prods - nn.JoinTable(3)
    local gmod = nn.gModule({q1, q2}, {qMult})
    return gmod
end
-------------------------------
-------------------------------
-- to get quaternion for rotation about x axis
-- input is {NX1}. Output is {NX4}
local function quatRx()
    local theta = -nn.Identity()
    local tb2 = theta - nn.MulConstant(0.5)
    local c_tb2 = tb2 - nn.TrigCosine()
    local s_tb2 = tb2 - nn.TrigSine() - nn.MulConstant(-1)
    local zero1 = theta - nn.MulConstant(0)
    local zero2 = theta - nn.MulConstant(0)
    
    local qx = {c_tb2, s_tb2, zero1, zero2} - nn.JoinTable(2)
    
    local gmod = nn.gModule({theta}, {qx})
    return gmod
end
-------------------------------
-------------------------------
-- to get quaternion for rotation about x axis
-- input is {NX1}. Output is {NX4}
local function quatRy()
    local theta = -nn.Identity()
    local tb2 = theta - nn.MulConstant(0.5)
    local c_tb2 = tb2 - nn.TrigCosine()
    local s_tb2 = tb2 - nn.TrigSine() - nn.MulConstant(-1)
    local zero1 = theta - nn.MulConstant(0)
    local zero2 = theta - nn.MulConstant(0)
    
    local qx = {c_tb2, zero1, s_tb2, zero2} - nn.JoinTable(2)
    
    local gmod = nn.gModule({theta}, {qx})
    return gmod
end
-------------------------------
-------------------------------
-- to get quaternion for rotation about x axis
-- input is {NX2}. Output is {NX4}
-- rot : r1 = Rx(-el)*Ry(-az)*invCorrMat
-- invCorrMat = {0.5,0.5,0.5,-0.5}
local function azelToQuatModule()
    local thetas = -nn.Identity()
    local az = thetas - nn.Narrow(2,1,1) 
    local el = thetas - nn.Narrow(2,2,1)
    
    local q_az = az - nn.MulConstant(-1) - quatRy() - nn.Unsqueeze(2) -- N X 1 X 4
    local q_el = el - nn.MulConstant(-1) - quatRx() - nn.Unsqueeze(2) -- N X 1 X 4

    local q_0pt5 = q_el - nn.MulConstant(0) - nn.AddConstant(0.5)
    local q_invCorr_pos = q_0pt5 - nn.Narrow(3,1,3)
    local q_invCorr_neg = q_0pt5 - nn.Narrow(3,4,1) - nn.MulConstant(-1)
    local q_invCorr = {q_invCorr_pos, q_invCorr_neg} - nn.JoinTable(3)
    
    local qRot = {q_el,q_az} - HamiltonProductModule()
    local q = {qRot,q_invCorr} - HamiltonProductModule() - nn.Sum(2) --basically squeeze !
    
    local gmod = nn.gModule({thetas}, {q})
    return gmod
end
-------------------------------
-------------------------------
-- input is BXPX4 quaternions, output is also BXPX4 quaternions
local function quatConjugateModule()
    local split = nn.ConcatTable():add(nn.Narrow(3,1,1)):add(nn.Narrow(3,2,3))
    local mult = nn.ParallelTable():add(nn.Identity()):add(nn.MulConstant(-1,false))
    local qc = nn.Sequential():add(split):add(mult):add(nn.JoinTable(3))
    return qc
end
-------------------------------
-------------------------------
-- input is {BXPX4 vectors, BXPX4 quaternions} output is BXPX3 rotated vectors
-- input vectors have 'real' dimension = 0
local function quatRotateModule()
    local quatIn = - nn.Identity()
    local quat = quatIn - nn.Contiguous()
    local vec = nn.Identity()()
    
    local quatConj = quatConjugateModule()(quat)
    local mult = HamiltonProductModule()({HamiltonProductModule()({quat,vec}),quatConj})
    local truncate = nn.Narrow(3,2,3)(mult)
    local gmod = nn.gModule({vec, quatIn}, {truncate})
    return gmod
end
-------------------------------
-------------------------------
local function quat2rot(q)
    local q = q:contiguous():view(4)
    local w = q[1]; local x = q[2]; local y = q[3]; local z = q[4];
    local xx2 = 2 * x * x
    local yy2 = 2 * y * y
    local zz2 = 2 * z * z
    local xy2 = 2 * x * y
    local wz2 = 2 * w * z
    local zx2 = 2 * z * x
    local wy2 = 2 * w * y
    local yz2 = 2 * y * z
    local wx2 = 2 * w * x

    local rmat = torch.Tensor(3,3):fill(0)
    rmat[1][1] = 1 - yy2 - zz2
    rmat[1][2] = xy2 - wz2
    rmat[1][3] = zx2 + wy2
    rmat[2][1] = xy2 + wz2
    rmat[2][2] = 1 - xx2 - zz2
    rmat[2][3] = yz2 - wx2
    rmat[3][1] = zx2 - wy2
    rmat[3][2] = yz2 + wx2
    rmat[3][3] = 1 - xx2 - yy2

    return rmat
end
-------------------------------
-------------------------------
local function rot2quat(rmat)
    local rmat = rmat:contiguous():view(3,3)
    local m00 = rmat[1][1]; local m01 = rmat[1][2]; local m02 = rmat[1][3];
    local m10 = rmat[2][1]; local m11 = rmat[2][2]; local m12 = rmat[2][3];
    local m20 = rmat[3][1]; local m21 = rmat[3][2]; local m22 = rmat[3][3];
    
    local tr = m00 + m11 + m22
    local qw, qx, qy, qz

    if (tr > 0) then
        local S = torch.sqrt(tr+1.0) * 2;
        qw = 0.25 * S;
        qx = (m21 - m12) / S;
        qy = (m02 - m20) / S;
        qz = (m10 - m01) / S;
    elseif ((m00 > m11) and (m00 > m22)) then
        local S = torch.sqrt(1.0 + m00 - m11 - m22) * 2;
        qw = (m21 - m12) / S;
        qx = 0.25 * S;
        qy = (m01 + m10) / S; 
        qz = (m02 + m20) / S; 
    elseif(m11 > m22) then
        local S = torch.sqrt(1.0 + m11 - m00 - m22) * 2;
        qw = (m02 - m20) / S;
        qx = (m01 + m10) / S; 
        qy = 0.25 * S;
        qz = (m12 + m21) / S; 
    else
        local S = torch.sqrt(1.0 + m22 - m00 - m11) * 2;
        qw = (m10 - m01) / S;
        qx = (m02 + m20) / S;
        qy = (m12 + m21) / S;
        qz = 0.25 * S;
    end
    
    local q = torch.Tensor({qw,qx,qy,qz})
    return q
end
-------------------------------
-------------------------------
-- Trying to figure out rotation distribution
-- corrMat = inv(angle2dcm(-pi/2, -pi/2, 0,'XYZ'))
-- corrMat = torch.Tensor({{0,0,-1}, {1,0,0}, {0,-1,0}})
-- [a1,a2,a3]=dcm2angle(var.extrinsic(1:3,1:3)*corrMat,'YXZ'); disp([-a1,-a2,a3]*180/pi) --have only tested with theta=0
-- So RotMat = angle2dcm([-az,-el,0])*inv(corrMat)
local corrMat = torch.Tensor({{0,0,-1}, {1,0,0}, {0,-1,0}})
local invCorrMat = torch.inverse(corrMat)

local function Rx(theta)
    return torch.Tensor({{1, 0, 0}, {0, torch.cos(theta), torch.sin(theta)}, {0, -torch.sin(theta), torch.cos(theta)}})
end

local function Ry(theta)
-- % rotation about x-axis by angle theta
    return torch.Tensor({{torch.cos(theta), 0, -torch.sin(theta)}, {0, 1, 0}, {torch.sin(theta), 0, torch.cos(theta)}})
end

local function Rz(theta)
-- % rotation about x-axis by angle theta
    return torch.Tensor({{torch.cos(theta), -torch.sin(theta), 0}, {torch.sin(theta), torch.cos(theta), 0}, {0, 0, 1}})
end

local function azel2rot(az, el)
    local r1 = torch.mm(Rx(-el*math.pi/180), Ry(-az*math.pi/180))
    return torch.mm(r1,invCorrMat)
end
-------------------------------
-------------------------------
local function quatRegressor(nC)
    return nn.Sequential():add(nn.Linear(nC,4)):add(nn.Normalize(2))
end

local function eulerPredToQuat(nC, offsetAz, elMin, elRange)
    local offsetAz = offsetAz or 0
    
    local feat =  - nn.Identity()
    local az = feat - nn.Linear(nC,1) - nn.Sigmoid() - nn.MulConstant(math.pi*2) - nn.AddConstant(offsetAz)
    az = az:annotate{name = 'az'}

    local el = feat - nn.Linear(nC,1) - nn.Sigmoid() - nn.MulConstant(elRange) - nn.AddConstant(elMin)
    el = el:annotate{name = 'el'}

    local quat = {az,el}  - nn.JoinTable(2) - azelToQuatModule() - nn.Unsqueeze(2) - quatConjugateModule() - nn.Sum(2)
    local gmod = nn.gModule({feat}, {quat})
    return gmod
end

local function quatPredSampleModule(nInpCh, nOptions, useEuler, elMin, elRange)
    
    local elMin = elMin or -math.pi/9
    local elRange = elRange or math.pi/3
    
    if(nOptions == 1) then
        return useEuler and eulerPredToQuat(nInpCh,0,elMin,elRange) or quatRegressor(nInpCh)
    end
    -- let K = nOptions
    local feat = -nn.Identity()
    -- probs = B X K
    local probs = feat - nn.Linear(nInpCh, nOptions) - nn.SoftMax()
    probs = probs:annotate{name = 'probs'}

    -- samples = B X K (one-hot)
    local samples = probs - nn.ReinforceCategorical(0.95)
    samples = samples:annotate{name = 'samples'}
    local sampleQuatRep = samples - nn.Replicate(4,2)
    
    local quatTable = {}
    for p=1,nOptions do
        local offsetAz = math.pi*2*p/nOptions
        quatTable[p] = feat - (useEuler and eulerPredToQuat(nInpCh,offsetAz,elMin,elRange) or quatRegressor(nInpCh)) - nn.Unsqueeze(3)
    end

    local quatAll = quatTable - nn.JoinTable(3)
    local quat = {quatAll, sampleQuatRep} - nn.CMulTable() - nn.Sum(3)
    quat = quat:annotate{name = 'quat'}
    
    local gmod = nn.gModule({feat}, {quat})
    return gmod
end
-------------------------------
-------------------------------
M.quat2rot = quat2rot
M.rot2quat = rot2quat
M.quatConjugateModule = quatConjugateModule
M.quatRotateModule = quatRotateModule
M.quatRx = quatRx
M.quatRy = quatRy
M.Rx = Rx
M.Ry = Ry
M.Rz = Rz
M.HamiltonProductModule = HamiltonProductModule
M.azel2rot = azel2rot
M.quatPredSampleModule = quatPredSampleModule
M.azelToQuatModule = azelToQuatModule
return M