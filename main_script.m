%% build a folder for data storation
pname='training_dataset/';
if ~exist(pname,'dir')
    mkdir(pname);
end

%% generate 2000 training dataset(reduce the number temperarily for testing)
training_dataset_generation(1:10,pname);

%% prepare for training
training_dataset_prepare(pname);

%% training process
net=unet3d_training(pname);

%% test
testfolder{1}=['inputsTest/'];
testfolder{2}=['labelsTest/'];
for ii=1
    volReader = @(x) matRead(x);
    volLoc=[pname testfolder{ii}];
    inputsTemp = imageDatastore(volLoc, 'FileExtensions','.mat','ReadFcn',volReader);
    inputsval=inputsTemp;
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
    inputsval.Files=Files;
end
for ii=2
    inputsName = dir([pname testfolder{ii} '*.mat']);
    for jj=1:length(inputsName)
        temp=load([inputsName(ii).folder '/' inputsName(ii).name]);
        labels(:,:,:,jj)=temp.image;
    end
end

load('net/nonorm-3_64_0.5/net_final.mat','net');
wmap_es = squeeze(predict(net, inputsval));
mean(abs(col(labels-wmap_es)))
figure,
subplot(1,2,1)
imagesc(labels(:,:,16,1),[-50 50]);colorbar;
subplot(1,2,2)
imagesc(wmap_es(:,:,16,1),[-50 50]);colorbar;

