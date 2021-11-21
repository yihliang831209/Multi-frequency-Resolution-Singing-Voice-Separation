clear all;
% yihliang modified。All sources are multiplied by uniform[0.7~1.0] gain. Vocals are processed by VAD first

p = genpath('Dev');

length_p = size(p,2);%Length of string p
path = {};%Create a cell array, each cell of the array contains a directory
temp = [];

for i = 1:length_p %Look for the delimiter';', once found, write the path temp into the path array
    if p(i) ~= ';'
        temp = [temp p(i)];
    else 
        temp = [temp '\']; %Add'\' at the end of the path
        path = [path ; temp];
        temp = [];
    end
end  


file_num = size(path,1); % Number of subfolders


for i = 2:(file_num)
    
    file_path =  path{i}; % Folder path
    wav_path_list = dir(strcat(file_path,'*.wav'));
    wav_num = length(wav_path_list); %Number of files in this folder
    if wav_num > 0
        file_name = wav_path_list(1).name;
        [x,fs] = audioread(strcat(file_path,file_name));
        tamp_instrument = x(:,1); 
        tamp_instrument = resample(tamp_instrument,16000,fs);
        tamp = zeros([length(tamp_instrument),1]);
        for j = 1:(wav_num-1)
            file_name = wav_path_list(j).name;
            [x,fs] = audioread(strcat(file_path,file_name));
            tamp_instrument = x(:,1)*unifrnd(0.7,1.0);  
            tamp_instrument = resample(tamp_instrument,16000,fs);
            tamp = tamp + tamp_instrument;
            fprintf('%d %d %s\n',i,j,strcat(file_path,file_name));% Display the path and audio file name being processed
        end
        music_ = tamp;
    if wav_num ==4
        file_name = wav_path_list(4).name;
        [x1,fs] = audioread(strcat(file_path,file_name));
        tamp_vocal = x1(:,1)*unifrnd(0.7,1.0);
        tamp_vocal = resample(tamp_vocal,16000,fs);
        rest_v = newVAD(tamp_vocal,16000);
        audiowrite(sprintf('vocal_100songs_16k_VAD/vocal_%s.wav',string(-1+i)),rest_v,16000);
        fprintf('%d %s\n',i,strcat(file_path,file_name));%Display the path and audio file name being processed
    else
        disp('smoe file dosent have 4 signal');
        break;
    end    
    audiowrite(sprintf('music_100songs_16k/instrument_%s.wav',string(-1+i)),music_,16000);
    disp('save sucess'+string(i))
    
    end
end

