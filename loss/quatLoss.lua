require 'nn'

local QuatCriterion, parent = torch.class('nn.QuatCriterion', 'nn.Criterion')

function QuatCriterion:updateOutput(input, target)
    local _input = input:view(-1,4)
    local _target = target:view(-1,4)
    local bs = _input:size(1)
    self.errPos = (_input - _target):norm(2,2):pow(2):view(-1)
    --print(self.errPos:size())
    self.errNeg = (_input + _target):norm(2,2):pow(2):view(-1)
    local errs = torch.cat(self.errPos, self.errNeg, 2):min(2)
    self.errs = errs:clone():mul(0.25)
    --print(_input:norm(2,2):min(), _input:norm(2,2):max(), self.errPos:mean()*0.5, self.errNeg:mean()*0.5)
    --print(self.errPos)
    --print(_input)
    --print(_target)
    local mod = nn.MSECriterion():cuda()
    --print(mod:forward(_input, _target), )
    self.output = errs:mean()*0.25
    return self.output
end

function QuatCriterion:updateGradInput(input, target)
    local _input = input:view(-1,4)
    local _target = target:view(-1,4)
    local bs = _input:size(1)
    self.gradInput = _input - _target
    local gradInputNeg = _input + _target
    for b=1,bs do
        if(self.errPos[b] > self.errNeg[b]) then
            self.gradInput[b]:copy(gradInputNeg[b])
        end
    end
    self.gradInput = self.gradInput:reshape(input:size()):mul(0.5/bs)
    return self.gradInput
end