local M = {}
-------------------------------
-------------------------------
-- returns Xs X Ys X Zs X 3
local function meshGrid(pointsX, pointsY, pointsZ)
    -- Xs, Ys, Zs = MeshGrid
    
    local gridSize = {}
    gridSize[1] = pointsX:size(1)
    gridSize[2] = pointsY:size(1)
    gridSize[3] = pointsZ:size(1)
    
    local xs = torch.repeatTensor(pointsX:view(-1, 1, 1, 1), 1, gridSize[2], gridSize[3], 1)
    local ys = torch.repeatTensor(pointsY:view(1, -1, 1, 1), gridSize[1], 1, gridSize[3], 1)
    local zs = torch.repeatTensor(pointsZ:view(1, 1, -1, 1), gridSize[1], gridSize[2], 1, 1)
    return torch.cat(xs, torch.cat(ys, zs))
end
-------------------------------
-------------------------------
M.meshGrid = meshGrid
return M