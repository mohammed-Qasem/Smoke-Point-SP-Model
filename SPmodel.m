% This script predicts the smoke point(SP)using artificial neural network 
% 10 features of the input  and one output are loaded and the network is
%  trained to generate a suitable prediction model 
% The Authors:
% 1. Mohammed Ameen Ahmed Qasem  
% 3. Abdul Gani Abdul Jameel   
% 4.Eid Al Mutairi      

%--------------------------------------------------------------------
clc;
chos = 0;
out  = 1;
NP=10;
possibility=5;
while chos~=possibility,
    chos=menu('MOHAMMED A.Q: Chemical  USING NEURAL NETWORK','Load Input File','Load Target File',... 
             'Create And Train','MyTraining','Predction','Exit');
%----------------------------------------------------------------------
    if chos==1, % This option for loading input file
         [namefile,pathname]=uigetfile('*.*','Select  Input File');
        if namefile~=0   % check if the input file is selected
            [x,map]=xlsread(strcat(pathname,namefile));
        else
            warndlg('Input File must be selected.',' Warning ')
        end
        disp('An Input  has just been loaded');
         
    end
%----------------------------------------------------------------------
    if chos==2,  % This option for loading target file
         [namefile,pathname]=uigetfile('*.*','Select Target File');
        if namefile~=0 % check if the target file is selected
            [t,map]=xlsread(strcat(pathname,namefile));
        else
            warndlg('Target File must be selected.',' Warning ')
        end
        disp('The Target  has just been loaded.');
    end
%----------------------------------------------------------------------
  if chos==3,% Create a nueral netwrok and Train
   [net] =  createnn(x',t')   
  end
  if chos==4, % This option will train the neural network after selecting the best parameters setting
   net=training(x,t)
  end
%----------------------------------------------------------------------
 if chos==5, % This option is selected to predict the output of TSI with respect to the input
 
     prompt={sprintf('%s','X1,X2,X3,X4,..... ')};
                title= ('Please enter the input');
                lines=1;
                def={'0,1,2,3,4,5,6,7,8,9'};
                answer=inputdlg(prompt,title,lines,def);
                zparameter=double(str2num(char(answer)));
                X =zparameter;
   
     Y = net(X');%predict(net, X)
      prompt={sprintf('%s','The predicted Output Y ')};
       title= ('Output ');
       lines=1;
        def={num2str(Y) };
         answer=inputdlg(prompt,title,lines,def);
 end
%----------------------------------------------------------------------
if chos==6, % exit
     break;
 end
end
function net=training(x,t)
x=x';
t=t';
 
trainFcn = 'trainbr';  % Bayesian Regularization backpropagation.

% Create a Fitting Network
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize,trainFcn);
net.trainParam.goal   = 0.00801;  
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

net.trainParam.mu=0.0265   %i;

% Train the Network
[net,tr] = train(net,x,t);
x_test = x(:, tr.testInd);

% Test the Network
y_test = net(x_test);
%y_test = y(tr.testInd);
y_test_real = t(tr.testInd);
e = gsubtract(t(tr.testInd),y_test);
performance = perform(net,t(tr.testInd),y_test)
view(net)
end
%--------------------------------------------------------------------------
function [net] = createnn(P,T)

alphabet = P;
targets  = T;

[R,Q]  = size(alphabet);
[S2,Q] = size(targets);
S1     = 100;
%S11    = 200;
 
trainFcn = 'trainbr'; 
% Create a Fitting Network
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize,trainFcn);
 %trainParam: .showWindow, .showCommandLine, .show, .epochs,
 %.time, .goal, .min_grad, .max_fail, .mu, .mu_dec,
 %.mu_inc, .mu_max

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
% traingda
%net = newff(minmax(alphabet),[S1 S11 S2],{'tansig' 'tansig' 'tansig'},'traingda');
net = newff(minmax(alphabet),[S1 S2],{'tansig' 'tansig'},'trainbr');
net.LW{2,1}           = net.LW{2,1}*0.01;
net.b{2}              = net.b{2}*0.01;
net.performFcn        = 'mse';
net.trainParam.goal   = 0.001;
net.trainParam.show   = 20;
net.trainParam.epochs = 500;
net.trainParam.mc     = 0.95;
P                     = alphabet;
T                     = targets;
[net,tr]              = train(net,P,T);
end
%--------------------------------------------------------------------------