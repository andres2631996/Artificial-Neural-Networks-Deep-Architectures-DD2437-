% Function approximation with MLP
close all
% Patterns and targets 


% Show function to approximate in a mesh plot
figure()
x=[-5:0.5:5]';
y=[-5:0.5:5]';
z=exp(-x.*x*0.1) * exp(-y.*y*0.1)' - 0.5;
mesh(x, y, z);


nx=size(x,1);
ny=size(y,1);
ndata=nx*ny;
targets = reshape (z, 1, ndata);
[xx, yy] = meshgrid (x, y);
patterns = [reshape(xx, 1, ndata); reshape(yy, 1, ndata)];
patterns_orig=patterns;

% Data shuffling
aux=[patterns;targets]; % Matrix with both patterns and targets
aux=aux(:,randperm(size(aux,2)));
patterns=aux(1:2,:); % Shuffled patterns
targets=aux(3,:); % Shuffled targets

% Add Gaussian noise to the training targets
snr=7.5;
targets=awgn(targets,snr);

% Validation data
validation_ratio=0.2;

patterns_val=patterns(:,1:round(validation_ratio*ndata));
targets_val=targets(:,1:round(validation_ratio*ndata));

% Training data
patterns_train=patterns(:,(round(validation_ratio*ndata)+1:end));
targets_train=targets(:,(round(validation_ratio*ndata)+1:end));




% Build the MLP

epochs=100; % Epochs to be applied
alpha=0.7; % Momentum coefficient
eta=0.03; % Learning rate

% Define weights for first and second layer
nodes1=10; % Number of nodes for the first layer
nodes2=1; % Number of nodes for the second layer
w=2*rand(nodes1,3)-1; % Weights for first layer
v=2*rand(nodes2,nodes1+1)-1; % Weights for second layer

error_train=zeros(epochs,1); % Training error
error_val=zeros(epochs,1); % Validation error

for i=1:epochs
    % Forward pass for training
    hin = w * [patterns_train ; ones(1,size(patterns_train,2))];
    hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,size(patterns_train,2))];
    oin = v * hout;
    out = 2 ./ (1+exp(-oin)) - 1;
    
    % Forward pass for validation
    hin_val = w * [patterns_val ; ones(1,size(patterns_val,2))];
    hout_val = [2 ./ (1+exp(-hin_val)) - 1 ; ones(1,size(patterns_val,2))];
    oin_val = v * hout_val;
    out_val = 2 ./ (1+exp(-oin_val)) - 1;
    
    
    % Error computation
    % For training
    mse_train=0.5*sum((out-targets_train).^2)/size(patterns_train,2);
    error_train(i,1)=mse_train;
    
    % For validation
    mse_val=0.5*sum((out_val-targets_val).^2)/size(patterns_val,2);
    error_val(i,1)=mse_val;
    
    if rem(i,10)==0
        fprintf('Epoch: %d / Training error: %f / Validation error: %f\n',i,mse_train,mse_val);
    end
    
    % Backward pass
    delta_o = (out - targets_train) .* ((1 + out) .* (1 - out)) * 0.5;
    delta_h = (v' * delta_o) .* ((1 + hout) .* (1 - hout)) * 0.5;
    delta_h = delta_h(1:nodes1, :);
    
    % Weight update
    if i==1
        dw = - (delta_h * [patterns_train ; ones(1,size(patterns_train,2))]');
        dv = - (delta_o * hout');
    else
        dw = (dw .* alpha) - (delta_h * [patterns_train ; ones(1,size(patterns_train,2))]') .* (1-alpha);
        dv = (dv .* alpha) - (delta_o * hout') .* (1-alpha);
    end
    w = w + dw .* eta;
    v = v + dv .* eta;
    
end

% Forward pass for all points
hin_all = w * [patterns_orig ; ones(1,size(patterns_orig,2))];
hout_all = [2 ./ (1+exp(-hin_all)) - 1 ; ones(1,size(patterns_orig,2))];
oin_all = v * hout_all;
out_all = 2 ./ (1+exp(-oin_all)) - 1;



figure()
% Plotting the evolution of the function
zz = reshape(out_all, nx, ny);
mesh(x,y,zz);
axis([-5 5 -5 5 -0.7 0.7]);


figure()
plot(1:epochs,error_train,'b',1:epochs,error_val,'r');
legend('Training','Validation')
xlabel('Epochs');
ylabel('Error');
title('Learning curve')

