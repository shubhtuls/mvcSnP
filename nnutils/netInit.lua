local M = {}

function M.weightsInit(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('Linear') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

function M.weightsInitPose(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('Linear') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end


function M.weightsZeroInit(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:fill(0.0)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:fill(0) end
      if m.bias then m.bias:fill(0) end
   end
end

function M.updateRewardsFunc(rewards)
    local function updateFunc(m)
        local name = torch.type(m)
        if name:find('Reinforce') then
            m:reinforce(rewards)
        end
    end
    return updateFunc
end

function M.setTestMode(m)
    local name = torch.type(m)
    if name:find('Reinforce') then
        m.testMode = true
    end
end

function M.unsetTestMode(m)
    local name = torch.type(m)
    if name:find('Reinforce') then
        m.testMode = false
    end
end

function M.nngraphExtractNode(gMod, name)
    for ix=1,#gMod.forwardnodes do
        if(gMod.forwardnodes[ix].data.annotations.name == name) then
            return gMod.forwardnodes[ix].data.module.output
        end
    end
end

return M