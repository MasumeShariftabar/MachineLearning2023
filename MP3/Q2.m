clc
clear
close all

%% data generation
n=900;

% preallocations
x=zeros(1,n);
dataset_1=zeros(n,7);

x(1,1:31)=1.3+0.2*rand;
for k=31:n-1
    x(1,k+1)=0.2*((x(1,k-30))/(1+x(1,k-30)^10))+0.9*x(1,k);
    dataset_1(k,2:6)=[x(1,k-3) x(1,k-2) x(1,k-1) x(1,k) x(1,k+1)];
end

dataset(1:600,2:6)=dataset_1(201:800,2:6);

t=1:600;

figure1 = figure('Color',[1 1 1]);
plot(t,x(201:800),'Linewidth',2)

[number_training,~]=size(dataset);
Rul=zeros(number_training/2,6);
Rules_total=zeros(number_training/2,6);

%% design fuzzy system

% s=1 --> 7 MF
% s=2 --> 14 MF
for s=1:2
    switch s
        case 1
        num_membership_functions=7;
        c=linspace(0.5,1.3,5);
        h=0.2;
        membership_functions=cell(num_membership_functions,2);
        for k=1:num_membership_functions
            if k==1
                membership_functions{k,1}=[0,0,0.3,0.5];
                membership_functions{k,2}='trapmf';
            elseif k==num_membership_functions 
                membership_functions{k,1}=[1.3,1.5,1.8,1.8];
                membership_functions{k,2}='trapmf';
            else
                membership_functions{k,1}=[c(k-1)-h,c(k-1),c(k-1)+h];
                membership_functions{k,2}='trimf';
            end
        end
        case 2
          num_membership_functions=15;
        c=linspace(0.3,1.5,13);
        h=0.1;
        membership_functions=cell(num_membership_functions,2);
        for k=1:num_membership_functions
            if k==1
                membership_functions{k,1}=[0,0,0.2,0.3];
                membership_functions{k,2}='trapmf';
            elseif k==num_membership_functions 
                membership_functions{k,1}=[1.5,1.6,1.8,1.8];
                membership_functions{k,2}='trapmf';
            else
                membership_functions{k,1}=[c(k-1)-h,c(k-1),c(k-1)+h];
                membership_functions{k,2}='trimf';
            end
        end    
    end
  %% assign degree to each rule
  
  vec_x=zeros(1,num_membership_functions);
  vec=zeros(1,5);
  for t=1:number_training
      dataset(t,1)=t;
        for i=2:6
            x=dataset(t,i);
                for j=1:num_membership_functions
                    if j==1
                    vec_x(1,j)=trapmf(x,membership_functions{1,1});
                    elseif j==num_membership_functions
                    vec_x(1,j)=trapmf(x,membership_functions{num_membership_functions,1});
                    else
                    vec_x(1,j)=trapmf(x,membership_functions{j,1}); 
                    end
                end
            [valu_x,column_x]=max(vec_x);
            vec(1,j-1)=max(vec_x);
            Rules(t,i-1)=column_x;
            Rules(t,6)=prod(vec);
            dataset(t,7)=prod(vec);
        end
  end
  %% delete extra rules
  
  
  rules_total(1,1:6)=Rules(1,1:6);
  i=1;
  for t=2:number_training
      m=zeros(1,i);
      for j=1:i
          m(1,j)=isequal(Rules(t,1:4),Rules_total(j,1:4));
          if m(1,j)==1 && Rules(t,6)>= Rules_total(j,6)
              Rules_total(j,1:6)=Rules(t,1:6);
          end
      end
      if sum(m) ==0
      Rules_total(i+1,1:6)=Rules(t,1:6);
      i=i+1;
      end
  end
  disp('*********************************************')
  disp(['Final rules for',num2str(num_membership_functions),'membership functions for each input variables'])
  final_Rules=Rules_total(1:i,:)
  
  %% create fuzzy inference system
  Fisname='Prediction controller';
  Fistype='mamdani';
  Andmethod='prod';
  Ormethod='max';
  Impmethod='prod';
  Agmethod='max';
  Defuzzmethod='centroid';
  fis=mamfis(Fisname,Fistype,Andmethod,Ormethod,Impmethod,Aggmethod,Defuzzmethod);
  %% add variables
  for num_input=1:4
      fis=addInput(fis,'input',['x',num2str(num_input)],[0.1 1.6]);
  end
  fis=addOutput(fis,'output','x5',[0.1 1.6]);
  %% add mf function
  for num_input=1:4
      for input_Rul=1:num_membership_functions
          fis=addMF(fis,'input',num_input,['A',num2str(input_Rul)],membership_functions{input_Rul,2},membership_functions{input_Rul,1});
      end
  end
  %% add rules
  fis_Rules=ones(i,7);
  fis_Rules(1:i,1:5)=Rules_total(1:i,1:5);
  fis=addrule(fis,fis_Rules);
  
  %% prediction of 300 points of chosen dataset
  
  table_prediction=zeros(300,2);
  f=1;
  for i=301:600
      input=dataset(i,2:6);
      output1=dataset(i,6);
  x5=evalfis([input(1,1);input(1,2);input(1,3);input(1,4)],fis);
  table_prediction(f,:)=[f,x5];
  f=f+1;
  end
  
  figure;
  plot(table_prediction(:,1),table_prediction(:,2),'r-.','Linewidth',2);
  hold on;
  plot(table_prediction(:,1),dataset(301:600,6),'b','Linewidth',2);
  legend('estimate value','real value')
end
