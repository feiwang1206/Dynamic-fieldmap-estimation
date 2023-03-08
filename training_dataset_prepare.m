function training_dataset_prepare(pname)
    ss=1:24;
    subject{1}='subject04_';
    subject{end+1}='subject05_';
    subject{end+1}='subject06_';
    subject{end+1}='subject18_';
    subject{end+1}='subject20_';
    subject{end+1}='subject38_';
    subject{end+1}='subject41_';
    subject{end+1}='subject42_';
    subject{end+1}='subject43_';
    subject{end+1}='subject44_';
    subject{end+1}='subject45_';
    subject{end+1}='subject46_';
    subject{end+1}='subject47_';
    subject{end+1}='subject48_';
    subject{end+1}='subject49_';
    subject{end+1}='subject50_';
    subject{end+1}='subject51_';
    subject{end+1}='subject52_';
    subject{end+1}='subject53_';
    subject{end+1}='subject54_';
    n=0;
    if ~exist([pname 'inputsTra'],'dir')
        mkdir([pname 'inputsTra']);
        mkdir([pname 'labelsTra']);
        mkdir([pname 'inputsVal']);
        mkdir([pname 'labelsVal']);
        mkdir([pname 'inputsTest']);
        mkdir([pname 'labelsTest']);        
    end
    %% validation dataset
    for subj=1
        for k=1:200
            if exist([pname subject{subj} 'recon_' num2str(k) '.mat'],'file')
                n=n+1;
                load([pname subject{subj} 'recon_' num2str(k) '.mat']);
                norm=max(abs(recon{1}(:)+recon{2}(:)))/2;
                clear image
                image(:,:,:,1)=real(recon{1}(:,:,ss))/norm;
                image(:,:,:,2)=real(recon{2}(:,:,ss))/norm;
                image(:,:,:,3)=imag(recon{1}(:,:,ss))/norm;
                image(:,:,:,4)=imag(recon{2}(:,:,ss))/norm;
                image(:,:,:,5)=recon{3}(:,:,ss)/1000;
                save([pname 'inputsVal/' subject{subj} num2str(k) '.mat'],'image');
                clear image
                image(:,:,:,1)=recon{6}(:,:,ss)/1;
                save([pname 'labelsVal/' subject{subj} num2str(k) '.mat'],'image');
           end
        end
    end
    %% test dataset
    for subj=2
        for k=1:200
            if exist([pname subject{subj} 'recon_' num2str(k) '.mat'],'file')
                n=n+1;
                load([pname subject{subj} 'recon_' num2str(k) '.mat']);
                norm=max(abs(recon{1}(:)+recon{2}(:)))/2;
                clear image
                image(:,:,:,1)=real(recon{1}(:,:,ss))/norm;
                image(:,:,:,2)=real(recon{2}(:,:,ss))/norm;
                image(:,:,:,3)=imag(recon{1}(:,:,ss))/norm;
                image(:,:,:,4)=imag(recon{2}(:,:,ss))/norm;
                image(:,:,:,5)=recon{3}(:,:,ss)/1000;
                save([pname 'inputsTest/' subject{subj} num2str(k) '.mat'],'image');
                clear image
                image(:,:,:,1)=recon{6}(:,:,ss)/1;
                save([pname 'labelsTest/' subject{subj} num2str(k) '.mat'],'image');
           end
        end
    end
    %% training dataset
    for subj=3:length(subject)
        for k=1:200
            if exist([pname subject{subj} 'recon_' num2str(k) '.mat'],'file')
                n=n+1;
                load([pname subject{subj} 'recon_' num2str(k) '.mat']);
                norm=max(abs(recon{1}(:)+recon{2}(:)))/2;
                clear image
                image(:,:,:,1)=real(recon{1}(:,:,ss))/norm;
                image(:,:,:,2)=real(recon{2}(:,:,ss))/norm;
                image(:,:,:,3)=imag(recon{1}(:,:,ss))/norm;
                image(:,:,:,4)=imag(recon{2}(:,:,ss))/norm;
                image(:,:,:,5)=recon{3}(:,:,ss)/1000;
                save([pname 'inputsTra/' subject{subj} num2str(k) '.mat'],'image');
                clear image
                image(:,:,:,1)=recon{6}(:,:,ss)/1;
                save([pname 'labelsTra/' subject{subj} num2str(k) '.mat'],'image');
           end
        end
    end
end
