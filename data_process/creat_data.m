clear all;
rng(1)
vocal_signal = zeros([0,1]);
music_signal = zeros([0,1]);
mix_signal = zeros([0,1]);

fs=16000;
v = 'vocal_100songs_16k_VAD//';
v_list=dir([v '*.wav']);

folder= 'music_100songs_16k//';
m_list=dir([folder '*.wav']);

file_count = 1;
fileLength = length(m_list);
rind= randperm(fileLength);
rid = randperm(fileLength);
vocal_portion = 0; % how much percent of training data is no vocal
train_data_TotalLength = 13.8;%5 Hr
win = 1-tukeywin(fs,0.25);
% plot(win)
% figure()
% plot(abs(fft(win)))
%%================= creat music/vocal paired data ===================
for i = 1:50
    disp('process'+string(i))
    
    vocalNames = ['.\vocal_100songs_16k_VAD\',v_list(i).name];
    musicNames = ['.\music_100songs_16k\',m_list(i).name];
    
    [y1,fs] = audioread(vocalNames);
    [y2,fs] = audioread(musicNames);
    temp_vocal = y1(:,1);
    temp_music = y2(:,1);
    % creat no vocal part
    % The calculation unit is based on one second
    len_voice = length(temp_vocal)/fs; % sec
    len_unvoice = fix(len_voice*vocal_portion); %Target unmanned length
    for j=1:1:len_unvoice %Silent for one second at a time
        start_point = fix(unifrnd(1,length(temp_vocal)-fs));
        temp_vocal(start_point:start_point+fs-1,1) = temp_vocal(start_point:start_point+fs-1,1) .* win;
    end
    if length(temp_vocal) < length(temp_music)
        len_ = fix(length(temp_vocal)/16000);
    else
        len_ = fix(length(temp_music)/16000);
    end
    
    rest_v = temp_vocal(1:len_*16000);
    rest_m = temp_music(1:len_*16000);
    
    
    temp_mix = rest_m + rest_v;
    audiowrite('16k_100percentVocal_scaled_randomMix\samples\music_'+string(i)+'.wav',rest_m,fs);
    audiowrite('16k_100percentVocal_scaled_randomMix\samples\vocal_'+string(i)+'.wav',rest_v,fs);
    audiowrite('16k_100percentVocal_scaled_randomMix\samples\mix_'+string(i)+'.wav',temp_mix,fs);
    
    music_signal = cat(1,music_signal,rest_m);
    vocal_signal = cat(1,vocal_signal,rest_v);
    mix_signal = cat(1,mix_signal,temp_mix);
    
    if  length(mix_signal)>fix(train_data_TotalLength*fs*3600/10)
        music_signal_reshape = reshape(music_signal,16000,1,[]);
        vocal_signal_reshape = reshape(vocal_signal,16000,1,[]);
        mix_signal_reshape = reshape(mix_signal,16000,[]);
        x=mix_signal_reshape;
        y=cat(2,vocal_signal_reshape,music_signal_reshape);
        save('16k_100percentVocal_scaled_randomMix/DSD100_16k_100percentVocal_pairedMix_randomMix_'+string(file_count)+'.mat','x','y');
        disp('export matfile!!!')
        vocal_signal = zeros([0,1]);
        music_signal = zeros([0,1]);
        mix_signal = zeros([0,1]);
        file_count = file_count+1;
    end
end
%%================= creat music/vocal paired data end ===================
%%================= random mix =================================
disp('paired mix finish, length = '+string(length(mix_signal)/(fs*3600))+'H' )
count = 1;
while 1
    for i = 1:50
        disp(string(count) + 'th times random mixing, length = '+string(length(mix_signal)/(fs*3600))+'H' )
        disp('process'+string(i))
        vocalNames = ['.\vocal_100songs_16k_VAD\',v_list(rind(i)).name];
        musicNames = ['.\music_100songs_16k\',m_list(rid(i)).name];
        
        [y1,fs] = audioread(vocalNames);
        [y2,fs] = audioread(musicNames);
        
        temp_vocal = y1(:,1).*unifrnd(0.7,1.0);
        temp_music = y2(:,1).*unifrnd(0.7,1.0);
        
        % creat no vocal part
        % The calculation unit is based on one second
        len_voice = length(temp_vocal)/fs; % sec
        len_unvoice = fix(len_voice*vocal_portion); %Target unmanned length
        for j=1:1:len_unvoice %Silent for one second at a time
            start_point = fix(unifrnd(1,length(temp_vocal)-fs));
            temp_vocal(start_point:start_point+fs-1,1) = temp_vocal(start_point:start_point+fs-1,1) .* win;
        end
        if length(temp_vocal) < length(temp_music)
            len_ = fix(length(temp_vocal)/16000);
        else
            len_ = fix(length(temp_music)/16000);
        end
        rest_v = temp_vocal(1:len_*16000);
        rest_m = temp_music(1:len_*16000);
        temp_mix = rest_m + rest_v;
        audiowrite('16k_100percentVocal_scaled_randomMix\samples\music_'+string(i+50*count)+'.wav',rest_m,fs);
        audiowrite('16k_100percentVocal_scaled_randomMix\samples\vocal_'+string(i+50*count)+'.wav',rest_v,fs);
        audiowrite('16k_100percentVocal_scaled_randomMix\samples\mix_'+string(i+50*count)+'.wav',temp_mix,fs);
        
        music_signal = cat(1,music_signal,rest_m);
        vocal_signal = cat(1,vocal_signal,rest_v);
        mix_signal = cat(1,mix_signal,temp_mix);
        if  length(mix_signal)>fix(train_data_TotalLength*fs*3600/10)
            music_signal_reshape = reshape(music_signal,16000,1,[]);
            vocal_signal_reshape = reshape(vocal_signal,16000,1,[]);
            mix_signal_reshape = reshape(mix_signal,16000,[]);
            x=mix_signal_reshape;
            y=cat(2,vocal_signal_reshape,music_signal_reshape);
            save('16k_100percentVocal_scaled_randomMix/DSD100_16k_100percentVocal_pairedMix_randomMix_'+string(file_count)+'.mat','x','y');
            disp('export matfile!!!')
            vocal_signal = zeros([0,1]);
            music_signal = zeros([0,1]);
            mix_signal = zeros([0,1]);
            file_count = file_count+1;
        end
        if file_count > 10
            break
        end
    end
    if file_count > 10
        break
    end
    count = count + 1;
end
%=========== creat validation data =================
rng(12)
v = 'vocal_100songs_16k_VAD//';
v_list=dir([v '*.wav']);

folder= 'music_100songs_16k//';
m_list=dir([folder '*.wav']);

file_count = 1;
fileLength = length(m_list);
rind= randperm(fileLength);
rid = randperm(fileLength);
fs = 16000;
vocal_signal = zeros([0,1]);
music_signal = zeros([0,1]);
mix_signal = zeros([0,1]);
vocal_portion = 0; % how much percent of training data is no vocal
win = 1-tukeywin(fs,0.25);
for i = 1:50
    disp('random mixing validation, length = '+string(length(mix_signal)/(fs*3600))+'H' )
    disp('process'+string(i))
    vocalNames = ['.\vocal_100songs_16k_VAD\',v_list(rind(i)).name];
    musicNames = ['.\music_100songs_16k\',m_list(rid(i)).name];
    
    [y1,fs] = audioread(vocalNames);
    [y2,fs] = audioread(musicNames);
    
    temp_vocal = y1(:,1).*unifrnd(0.7,1.0);
    temp_music = y2(:,1).*unifrnd(0.7,1.0);
    
    % creat no vocal part
    % The calculation unit is based on one second
    len_voice = length(temp_vocal)/fs; % sec
    len_unvoice = fix(len_voice*vocal_portion); %Target unmanned length
    for j=1:1:len_unvoice %Silent for one second at a time
        start_point = fix(unifrnd(1,length(temp_vocal)-fs));
        temp_vocal(start_point:start_point+fs-1,1) = temp_vocal(start_point:start_point+fs-1,1) .* win;
    end
    if length(temp_vocal) < length(temp_music)
        len_ = fix(length(temp_vocal)/16000);
    else
        len_ = fix(length(temp_music)/16000);
    end
    rest_v = temp_vocal(1:len_*16000);
    rest_m = temp_music(1:len_*16000);
    temp_mix = rest_m + rest_v;
    audiowrite('16k_100percentVocal_scaled_randomMix\samples_valid\music_'+string(i)+'.wav',rest_m,fs);
    audiowrite('16k_100percentVocal_scaled_randomMix\samples_valid\vocal_'+string(i)+'.wav',rest_v,fs);
    audiowrite('16k_100percentVocal_scaled_randomMix\samples_valid\mix_'+string(i)+'.wav',temp_mix,fs);
    
    music_signal = cat(1,music_signal,rest_m);
    vocal_signal = cat(1,vocal_signal,rest_v);
    mix_signal = cat(1,mix_signal,temp_mix);
    if  length(mix_signal)>fix(3000*fs)
        music_signal_reshape = reshape(music_signal,16000,1,[]);
        vocal_signal_reshape = reshape(vocal_signal,16000,1,[]);
        mix_signal_reshape = reshape(mix_signal,16000,[]);
        x=mix_signal_reshape;
        y=cat(2,vocal_signal_reshape,music_signal_reshape);
        save('16k_100percentVocal_scaled_randomMix/DSD100_16k_100percentVocal_pairedMix_randomMix_validation.mat','x','y');
        disp('export matfile!!!')
        vocal_signal = zeros([0,1]);
        music_signal = zeros([0,1]);
        mix_signal = zeros([0,1]);
        file_count = file_count+1;
        break
    end
end





