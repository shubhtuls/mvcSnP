require 'nn'

local TrigCosine, parent = torch.class("nn.TrigCosine", "nn.Module")
function TrigCosine:updateOutput(input)
    self.output = torch.cos(input)
    return self.output
end
function TrigCosine:updateGradInput(input, gradOutput)
    self.gradInput = torch.sin(input):mul(-1):cmul(gradOutput)
    return self.gradInput
end

local TrigSine, parent = torch.class("nn.TrigSine", "nn.Module")
function TrigSine:updateOutput(input)
    self.output = torch.sin(input)
    return self.output
end
function TrigSine:updateGradInput(input, gradOutput)
    self.gradInput = torch.cos(input):cmul(gradOutput)
    return self.gradInput
end