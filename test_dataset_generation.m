function recon=test_dataset_generation(rang,pname)
    addpath mirt
    setup;
    addpath SphericalHarmonics/coreFunctions
    addpath codes
    addpath(genpath('mreg_recon_tool'));
    scale=[1 1 1]*1;
    iter=20;
    if ~exist(pname,'dir')
        mkdir(pname);
    end

    pp='resource/';
    clear subject
    subject{1}='subject04_';

    dim=[64 64 50];
    data=load([pp 'data.mat']);
    data=data.data;
    smaps=double(data.smaps)/1024;
    trajectory=data.trajectory;
    traj = trajectory.trajectory;
    traj_idx = trajectory.idx;
    traj{1} = traj{1}(traj_idx{1},:);   
    traj{2} = traj{2}(traj_idx{2},:); 

    traj1{1} = [traj{1}(:,2), -traj{1}(:,1) , traj{1}(:,3)];
    traj1{2} = [traj{2}(:,2), -traj{2}(:,1) , traj{2}(:,3)];


    lengthP = 0;
    P = cell(1,lengthP);
    counter = 1;
    for k=1:1
        operator(1).handle = @identityOperator;
        operator(1).args = {};
        P{counter} = @L2Norm;%recon_details.penalty(k).norm;
        counter = counter + 1;
        P{counter} = 0.2;
        counter = counter + 1;
        for n=1:length(operator)
            P{counter} = operator(n).handle(operator(n).args{:});
            counter = counter + 1;
        end
    end

    xg=repmat((-32:31)',[1 64 50]);
    yg=repmat((-32:31),[64 1 50]);
    zg=repmat(permute((-25:24)',[2 3 1]),[64 64 1]);
    if gpuDeviceCount>0
        xg=gpuArray(xg);yg=gpuArray(yg);zg=gpuArray(zg);
    end

    for subj=1:length(subject)
        coil_num=16;
        smaps_l1=imresize4D(smaps,dim);
        smaps_s1=imresize4D(smaps,dim./scale);
        if gpuDeviceCount
            smaps_l1=gpuArray(smaps_l1);
            smaps_s1=gpuArray(smaps_s1);
        end


        for n=rang
            fname=[pname subject{subj} 'recon_' num2str(n) '.mat'];
            if ~exist(fname,'file')
                disp(num2str(n))
                cT1(1)=1.5*(rand*0.2+0.9);cT1(2)=1*(rand*0.2+0.9);cT1(3)=4.5*(rand*0.2+0.9);cT1(4)=1.4*(rand*0.2+0.9);
                cT2Star(1)=0.05*(rand*0.2+0.9);cT2Star(2)=0.03*(rand*0.2+0.9);cT2Star(3)=0.7*(rand*0.2+0.9);cT2Star(4)=0.015*(rand*0.2+0.9);
                cRho(1)=0.6*(rand*0.2+0.9);cRho(2)=0.7*(rand*0.2+0.9);cRho(3)=0.9*(rand*0.2+0.9);cRho(4)=0.6*(rand*0.2+0.9);

                image=load([pp subject{subj} 'image.mat']);
                image=double(image.image)/1024;
                bck=image(:,:,:,5)+128;
                bck=255-bck;
                clear VObj
                for k=1:dim(3)
                    for j=1:dim(2)
                        for i=1:dim(1)
                            tem=col(image(i,j,k,1:4));
                            tem=tem'/sum(tem);
                            tem2=tem;
                            tem2(isnan(tem))=0;
                            VObj.T1(i,j,k)=sum(cT1.*tem2);
                            VObj.T2Star(i,j,k)=sum(cT2Star.*tem2);
                            VObj.Rho(i,j,k)=sum(cRho.*tem2);
                        end
                    end
                end

                rho=imresize3D(VObj.Rho,dim);
                T1=imresize3D(VObj.T1,dim);
                TR=0.1;
                alpha=25;
                te=0.002;
                dt=5e-6;
                T1a=(1-exp(-TR./T1))./(1-cos(alpha/180*pi)*exp(-TR./T1));
                T1a((T1==0))=0;

                T2Star=imresize3D(VObj.T2Star,dim);

                rho1=rho;
                T1a1=T1a;
                T2Star1=T2Star;
                bck1=bck;
                if gpuDeviceCount>0
                    rho1=gpuArray(rho1);
                    T2Star1=gpuArray(T2Star1);
                    T1a1=gpuArray(T1a1);
                end
        %%
                mask=zeros(dim);
                bck1s=smooth3(bck1,'box',3);
                mask(bck1s>0.2*max(bck1s(:)))=1;

            %%     wmap0    
                dim2=dim*5;
                mask2=zeros(dim2);
                mask2(dim(1)*2+(1:dim(1)),dim(2)*2+(1:dim(2)),dim(3)*2+(1:dim(3)))=mask;
                for ii=1:dim2(1)
                    for jj=1:dim2(2)
                        mask2(ii,jj,1:(dim(3)*2))=mask2(ii,jj,2*dim(3)+1);
                    end
                end
            %% shimming simulation
                px=repmat([-dim2(1)/2:(dim2(1)/2-1)]',[1 dim2(2) dim2(3)]);
                py=repmat([-dim2(2)/2:(dim2(2)/2-1)],[dim2(1) 1 dim2(3)]);
                pz=repmat(permute([-dim2(3)/2:(dim2(3)/2-1)]',[3 2 1]),[dim2(1) dim2(2) 1]);
                pxyz=pz.^2./(px.^2+py.^2+pz.^2);
                B_temp=fftshift(1/3-pxyz).*fftn(((-9e-6)*mask2+0.36e-6*(1-mask2)));
                B_temp(isnan(B_temp))=0;
                wmap0=2*pi*3*42.58*10^6*ifftn(B_temp);
                wmap0=wmap0(dim(1)*2+(1:dim(1)),dim(2)*2+(1:dim(2)),dim(3)*2+(1:dim(3)));

                degreeMax = 2;
                orderMax = 2*degreeMax; % We calculate the same number of order as degree.You do not have to
                rhoReference = 0.5; % set the reference radius to 1 meter
                rk = createTargetPointGaussLegendreAndRectangle7(rhoReference,degreeMax,orderMax);
                bc(1).coefficient = zeros(degreeMax+1,orderMax+1);
                for i=1:length(rk)
                    po=floor((rk(i,1:3)+1).*dim/2)+1;
                    B(i)=wmap0(po(1),po(2),po(3));
                end
                [bc] = getSphericalHarmonicsCoefficientMeasure(B,degreeMax,orderMax,rk,'sch');
                x=((-dim(1)/2):(dim(1)/2-1))/(dim(1)/2);
                y=((-dim(2)/2):(dim(2)/2-1))/(dim(2)/2);
                z=((-dim(3)/2):(dim(3)/2-1))/(dim(3)/2);
                wmap01  = RebuildField(bc,rhoReference,x,y,z,'sch');
                wmap0=smooth3((wmap0-wmap01).*mask);
                %% phase
                map=smooth3(randn(dim*3),'box',15);
                map=smooth3(map,'box',15);
                map=map(dim(1)+(1:dim(1)),dim(2)+(1:dim(2)),dim(3)+(1:dim(3)));
                map=map/max(abs(map(:)));
                phase0=-1i*randn*map*2*pi;

                %% wmap error
                map=smooth3(randn(dim*3),'box',15);
                map=smooth3(map,'box',15);
                map=map(dim(1)+(1:dim(1)),dim(2)+(1:dim(2)),dim(3)+(1:dim(3)));
                map=map/max(abs(map(:)));
                wmap_err=smooth3(2*(rand-0.5)*100*map.*mask,'box',5);
    %%
                wmap_real=imresize3D(wmap0,dim);
                wmap_fake=imresize3D(wmap0+wmap_err,dim./scale);
                if gpuDeviceCount
                    wmap_real=gpuArray(wmap_real);
                    wmap_fake=gpuArray(wmap_fake);
                end

    %% rawdata
                rawdata{1}=zeros(length(traj1{1}),coil_num);
                rawdata{2}=zeros(length(traj1{2}),coil_num);
                for i=1:length(traj1{1})
                    if abs(traj1{1}(i,1))<pi/scale(1) && abs(traj1{1}(i,2))<pi/scale(2) && abs(traj1{1}(i,3))<pi/scale(3)
                        t=te+dt*i;
                        t2=exp(-t./T2Star1);
                        t2(T2Star1<=0)=0;
                        phase1=-1i*(traj1{1}(i,1)*xg+traj1{1}(i,2)*yg+traj1{1}(i,3)*zg);
                        phase2=1i*(wmap_real*t);
                        cons=repmat(rho1.*T1a1.*t2,[1 1 1 coil_num]).*smaps_l1;
    %                     cons=repmat(rho1,[1 1 1 coil_num]).*smaps_l1;

                        signal=cons.*exp(phase0+phase1+phase2)/sqrt(dim(1)*dim(2)*dim(3));
                        rawdata{1}(i,:)=sum(reshape(signal,[dim(1)*dim(2)*dim(3) coil_num]));
                        signal=cons.*exp(phase0-phase1+phase2)/sqrt(dim(1)*dim(2)*dim(3));
                        rawdata{2}(i,:)=sum(reshape(signal,[dim(1)*dim(2)*dim(3) coil_num]));
                    else
                        rawdata{1}(i,:)=0;
                        rawdata{2}(i,:)=0;
                    end
                end
                for in=1:size(rawdata{1},2)
                    rawdata{1}(:,in)=awgn(rawdata{1}(:,in),50);            
                    rawdata{2}(:,in)=awgn(rawdata{2}(:,in),50);
                end
                recon{7}=rawdata;


                traj2{1}(:,1)=traj{1}(:,1)*scale(1);
                traj2{1}(:,2)=traj{1}(:,2)*scale(2);
                traj2{1}(:,3)=traj{1}(:,3)*scale(3);
                subset1=(1:length(traj2{1}));
                Tt1=(subset1-1)*dt;


                traj2{2}(:,1)=traj{2}(:,1)*scale(1);
                traj2{2}(:,2)=traj{2}(:,2)*scale(2);
                traj2{2}(:,3)=traj{2}(:,3)*scale(3);
                subset2=(1:length(traj2{2}));
                Tt2=(subset2-1)*dt;
                Fg1=orc_segm_nuFTOperator_multi_sub({traj2{1}},dim./scale,smaps_s1,wmap_fake,dt,10,{Tt1+te});
                Fg2=orc_segm_nuFTOperator_multi_sub({traj2{2}},dim./scale,smaps_s1,wmap_fake,dt,10,{Tt2+te});


                [recon{1}] = gather(regularizedReconstruction(Fg1,rawdata{1}, P{:},...
                    'tol',1e-5, ...
                    'maxit',iter, ...
                    'verbose_flag', 0));
                [recon{2}] = gather(regularizedReconstruction(Fg2,rawdata{2}, P{:},...
                            'tol',1e-5, ...
                            'maxit',iter, ...
                            'verbose_flag', 0));
                Fg=orc_segm_nuFTOperator_multi_sub(traj2,dim./scale,smaps_s1,wmap_fake,dt,10,{Tt1+te,Tt2+te});
                [recon{4}] = gather(regularizedReconstruction(Fg,[rawdata{1};rawdata{2}], P{:},...
                            'tol',1e-5, ...
                            'maxit',iter, ...
                            'verbose_flag', 0));

                recon{6}=gather(imresize3D(wmap_err,dim./scale));
                recon{3}=gather(wmap_fake);

                save(fname,'recon');
            end
        end
    end
