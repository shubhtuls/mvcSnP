local params = {}
params.expSetName = 'unsup'
params.benchmarkName = 'align'
params.evalSet = 'val'
params.classes = 'all'

for k,v in pairs(params) do params[k] = tonumber(os.getenv(k)) or os.getenv(k) or params[k] end

local synsets = {chair='03001627', aero='02691156', car='02958343'}
if(params.classes == 'all') then params.classes = {'aero','car','chair'} else params.classes = {params.classes} end

local netNames = {}
local netIter

-----------------------
-----------------------
if (params.expSetName == 'unsuprot') then
    netNames = {
        'depth_poseRegUnsup_nds80_np8_euler_prior1pt0_nc3',
        'mask_poseRegUnsup_nds80_np8_euler_prior0pt1_nc3'
    }
    netIter = 80000
end
if (params.expSetName == 'unsuprottrans') then
    netNames = {
        'depth_transRotRegUnsup_nds80_np8_euler_prior_nc3',
        'mask_transRotRegUnsup_nds80_np8_euler_prior_nc3'
    }
    netIter = 80000
end
if (params.expSetName == '3dsup') then
    netNames = {
        '3dSup'
    }
    netIter = 80000
end
if (params.expSetName == 'posesup') then
    netNames = {
        'depth_poseSup_nds80_nc3',
        'mask_poseSup_nds80_nc3'
    }
    netIter = 80000
end

if (params.expSetName == 'posepred') then
    netNames = {
        'posePred_np1_euler_nc3'
    }
    netIter = 80000
end

-----------------------
-----------------------

local benchmarkFileName
if(params.benchmarkName == 'shape') then
    benchmarkFileName = 'synthetic/shape.lua'
elseif(params.benchmarkName == 'align') then
    benchmarkFileName = 'synthetic/computeShapeAlignment.lua'
elseif(params.benchmarkName == 'pose') then
    benchmarkFileName = 'synthetic/rotation.lua'
end

-----------------------
-----------------------
for cx = 1,#params.classes do
    local class = params.classes[cx]
    for nx = 1,#netNames do
        jobStr = string.format('evalSet=%s netIter=%d synset=%s netName=%s_%s th %s \n', params.evalSet, netIter, synsets[class], class, netNames[nx], benchmarkFileName)

        if params.benchmarkName == 'align' then
            if params.expSetName == 'unsuprottrans' then
                jobStr = 'updateTrans=1 ' .. jobStr
            else
                jobStr = 'updateTrans=0 ' .. jobStr
            end
        end

        print(jobStr)
    end
end
