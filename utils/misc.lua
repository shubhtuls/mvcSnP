local M = {}
-------------------------------
-------------------------------
local function addTableRec(t1,t2,w1,w2)
    if(torch.isTensor(t1)) then
        return w1*t1+w2*t2
    end
    local result = {}
    for ix=1,#t1 do
        result[ix] = addTableRec(t1[ix],t2[ix],w1,w2)
    end
    return result
end
-------------------------------
-------------------------------
M.addTableRec = addTableRec
return M