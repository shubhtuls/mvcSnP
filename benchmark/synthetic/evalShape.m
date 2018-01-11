expSetNames = {'3dsup', 'posesup', 'unsuprot', 'unsuprottrans'};
%expSetNames = {'unsuprottrans'};
synsets = {'aero', 'car', 'chair'};
basedir = pwd();
resultsDir = fullfile(basedir, '..', '..', 'cachedir', 'resultsDir', 'shape');

for ex = 1:length(expSetNames)
    expSetName = expSetNames{ex};

    if(strcmp(expSetName,'unsuprot'))
        netNames = {
            'mask_poseRegUnsup_nds80_np8_euler_prior0pt1_nc3',
            'depth_poseRegUnsup_nds80_np8_euler_prior1pt0_nc3'
        };
        netIter = 80000;
    end

    if(strcmp(expSetName,'unsuprottrans'))
        netNames = {
            'depth_transRotRegUnsup_nds80_np8_euler_prior_nc3',
            'mask_transRotRegUnsup_nds80_np8_euler_prior_nc3'
        };
        netIter = 80000;
    end

    if(strcmp(expSetName, '3dsup'))
        netNames = {
            '3dSup'
        };
        netIter = 80000;
    end

    if(strcmp(expSetName, 'posesup'))
        netNames = {
            'mask_poseSup_nds80_nc3',
            'depth_poseSup_nds80_nc3'
        };
        netIter = 80000;
    end

    perfs = zeros(length(netNames), length(synsets));
    threshesOpt = zeros(length(netNames), length(synsets));
    for nx=1:length(netNames)
        for sx = 1:length(synsets)
            evalDirVal = fullfile(resultsDir, 'shapenet', [synsets{sx} '_' netNames{nx} '_' num2str(netIter) '_val']);
            evalDirTest = fullfile(resultsDir, 'shapenet', [synsets{sx} '_' netNames{nx} '_' num2str(netIter) '_test']);
            [iouVal, threshOpt] = iouBenchmark(evalDirVal,0.01:0.01:0.99);
            iouTest = iouBenchmark(evalDirVal,[threshOpt]);
            perfs(nx,sx) = iouTest;
            threshesOpt(nx,sx) = threshOpt;
        end
    end

    disp(perfs);
    save(fullfile(resultsDir, expSetName), 'perfs','netNames','synsets','threshesOpt');
end
