function net=unet3d_training(pname)
    %% training parameters
    epoch=10;
    MiniBatchSize=20;
    depth=3;
    n_filter=64;
    imageSize=[32 32 24 5];
    lr(1)=1e-3;
    droprate=0.5;


    %% read training dataset
    trainfolder{1}=['/inputsTra'];
    trainfolder{2}=['/labelsTra'];
    for ii=1:length(trainfolder)
        volReader = @(x) matRead(x);
        volLoc=[pname trainfolder{ii}];
        inputsTemp = imageDatastore(volLoc, 'FileExtensions','.mat','ReadFcn',volReader);
        inputs{ii}=inputsTemp;
        Files=[];
        nn=0;
        for jj=1:length(inputsTemp.Files)
            temp=inputsTemp.Files{jj,1};
            for kk=length(temp):-1:1
                if strcmp(temp(kk),'/')
                    temp2=temp(kk+1:(end-4));
                    break
                end
            end
            nn=nn+1;
            Files{nn,1}=inputsTemp.Files{jj,1};
        end
        inputs{ii}.Files=Files;
    end


    trainfolder{1}=['/inputsVal'];
    trainfolder{2}=['/labelsVal'];
    for ii=1:length(trainfolder)
        volReader = @(x) matRead(x);
        volLoc=[pname trainfolder{ii}];
        inputsTemp = imageDatastore(volLoc, 'FileExtensions','.mat','ReadFcn',volReader);
        inputsval{ii}=inputsTemp;
        Files=[];
        nn=0;
        for jj=1:length(inputsTemp.Files)
            temp=inputsTemp.Files{jj,1};
            for kk=length(temp):-1:1
                if strcmp(temp(kk),'/')
                    temp2=temp(kk+1:(end-4));
                    break
                end
            end
            nn=nn+1;
            Files{nn,1}=inputsTemp.Files{jj,1};
        end
        inputsval{ii}.Files=Files;
    end

    lgraph = unet3dLayers_nonorm(imageSize,depth,'NumFirstEncoderFilters',n_filter);
    CheckpointPath=['net/nonorm-' num2str(depth) '_' num2str(n_filter) '_' num2str(droprate) '/'];
    mkdir(CheckpointPath);

    patchds = randomPatchExtractionDatastore(inputsval{1},inputsval{2},imageSize(1:3),'PatchesPerImage',1);
    options = trainingOptions('adam', ...
        'InitialLearnRate',lr, ...
        'LearnRateSchedule','piecewise',...
        'LearnRateDropFactor',droprate,...
        'Shuffle','every-epoch',...
        'MaxEpochs',epoch, ...
        'VerboseFrequency',floor(length(inputs{1}.Files)/MiniBatchSize),...
        'MiniBatchSize',MiniBatchSize,...
        'Plots','training-progress',...
        'ValidationFrequency',floor(length(inputs{1}.Files)/MiniBatchSize),...
        'ValidationData',patchds);
    patchds1 = randomPatchExtractionDatastore(inputs{1},inputs{2},imageSize(1:3),'PatchesPerImage',1);
    if exist([CheckpointPath '/net_final.mat'],'file')
        load([CheckpointPath '/net_final.mat'],'net');
        [net,info] = trainNetwork(patchds1,layerGraph(net),options);
    else
        [net,info] = trainNetwork(patchds1,lgraph,options);
    end
    save([CheckpointPath '/net_final.mat'],'net');
    save([CheckpointPath '/info.mat'],'info');
end
