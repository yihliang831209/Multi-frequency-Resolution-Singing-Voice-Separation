function [rest_s,new_s] = newVAD(s,fs)
FrameSize = fs*0.032; % frame size 32ms
ShiftSize = fs*0.016; % shift 16ms
Overlap = FrameSize-ShiftSize;
multiple_number=0.2; %0.2
s_temp = [];
temp = [];
temp_all = [];
new = [];
rest_s = [];
t = s;
time=0;
%% frame size 32ms / shift 16ms (overlap 16ms)
for i=FrameSize+1:ShiftSize:ShiftSize*(floor(length(s)/ShiftSize))+1  %0.1s
    time=time+1;
    temp = log(norm(t(i-FrameSize:i-1))/norm(t) +0.0001);
    temp_all = [temp_all;temp];
end

min_temp = min(temp_all);
threshold_range = max(temp_all)-min_temp;
predict_threshold = threshold_range*multiple_number+min_temp;

for i=1:time
    if temp_all(i)>predict_threshold
        new = [new;1*ones(ShiftSize,1)];
    else
        new = [new;0*ones(ShiftSize,1)];
    end
end
s_temp(1:ShiftSize*(floor(length(s)/ShiftSize)),1) = s(1:ShiftSize*(floor(length(s)/ShiftSize)),1);
s_temp = s_temp(Overlap+1:length(s_temp));
new_s = new.*s_temp;
for j=1:length(new)
    if new(j)==1
        rest_s = [rest_s;new_s(j)];
    end
end