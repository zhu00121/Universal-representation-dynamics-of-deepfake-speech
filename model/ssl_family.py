import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
from transformers import AutoModel, AutoProcessor
from transformers import AutoFeatureExtractor, WavLMModel
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
from speechbrain.processing.features import STFT


###########

class modulator(nn.Module):
    def __init__(
        self,
        sample_rate,
        win_length,
        hop_length,
        *args,
        **kwargs):

        super().__init__(*args,**kwargs)
        
        # short time fouriere transform
        self.compute_STFT = STFT(sample_rate=sample_rate, win_length=win_length, hop_length=hop_length) 

    def batch_fft(self, x):
        """
        input_shape -> (num_bacth, time, channels)
        """
        eps = 1e-10
        x_mod = torch.abs(self.compute_STFT(x)) # (num_batch, time, num_mod_freq, 2, num_features)
        x_mod = x_mod.pow(2).sum(-2) # combine real and imaginary parts
        x_mod_ave = torch.mean(x_mod,axis=1) # average over time axis -> (num_batch, num_mod_freq, num_features)
        x_mod_ave = torch.log(x_mod_ave+eps) # take log of power
        return x_mod_ave
    
    def forward(self,x):
        assert x.ndim == 3, "input representation needs to be 3D (batch, time, feature_channel)"
        ot = self.batch_fft(x)
        return ot

#########

class SSL_diagnoser_mod(nn.Module):
    def __init__(
        self,
        encoder_choice: str = 'wavlm',
        ssl_encoder_source: str = "microsoft/wavlm-base-plus",
        num_ssl_feat: int = 768,
        num_fc_neurons:int=-1,
        num_classes: int = 1,
        freeze_encoder: bool = True,
        pooling_1: str = 'atn',
        pooling_2: str = 'atn',
        sample_rate: int = 50,
        win_length: int = 128,
        hop_length: int = 32,
        dp = 0.25,
        *args,
        **kwargs
        ):
        
        super().__init__(*args, **kwargs)

        self.processor = AutoFeatureExtractor.from_pretrained(ssl_encoder_source)
        self.feature_extractor = WavLMModel.from_pretrained(ssl_encoder_source)

        for param in self.feature_extractor.parameters():
            param.requires_grad = not freeze_encoder

        num_input_feat_1 = num_ssl_feat*2 if pooling_1 == 'atn' else num_ssl_feat
        num_input_feat_2 = num_ssl_feat*2 if pooling_2 == 'atn' else num_ssl_feat
        if num_fc_neurons == -1: num_fc_neurons = num_input_feat_1 + num_input_feat_2

        self.fc = nn.Sequential(
            nn.Linear(num_input_feat_1 + num_input_feat_2, num_fc_neurons),
            nn.Dropout(p=dp),
            nn.LeakyReLU(0.1),
            nn.Linear(num_fc_neurons, num_classes)
        )

        self.weights_stack = nn.Parameter(torch.ones(self.feature_extractor.config.num_hidden_layers))

        self.compute_STFT = STFT(sample_rate=sample_rate, win_length=win_length, hop_length=hop_length) # 256ms window, with 64ms hop length
        
        if pooling_1 == "avg":
            self.pooling_1 = lambda x: F.adaptive_avg_pool1d(x, 1)
        elif pooling_1 == 'atn':
            self.pooling_1 = AttentiveStatisticsPooling(
                num_ssl_feat,
                attention_channels=num_ssl_feat,
                global_context=True
            )
        if pooling_2 == "avg":
            self.pooling_2 = lambda x: F.adaptive_avg_pool1d(x, 1)
        elif pooling_2 == 'atn':
            self.pooling_2 = AttentiveStatisticsPooling(
                num_ssl_feat,
                attention_channels=num_ssl_feat,
                global_context=True
            )

    def forward(self, x):
        # Preprocessing the data
        input_values = self.processor(x, sampling_rate=16000, return_tensors="pt").input_values[0]
        input_values = input_values.to(device=x.device, dtype=x.dtype)

        # Extract wav2vec2 hidden states and perform a weighted sum
        features = self.feature_extractor(input_values, output_hidden_states=True)
        features = self.weighted_sum(features.hidden_states[1:]) # (batch, time, features)

        # Calculate modulation energies
        mod_features = self.mod_fft(features)
        mod_features = mod_features.permute(0, 2, 1) # (batch, features, mod_freq)
        mod_features = self.pooling_2(mod_features).squeeze(-1) # (batch, features)

        # Calculate temporal statistics
        temporal_features = features.permute(0, 2, 1) # (batch, time, features )=> (batch, features, time)
        temporal_features = self.pooling_1(temporal_features).squeeze(-1) # (batch, features)

        features_final = torch.cat((temporal_features, mod_features),axis=1) # concat features
        output = self.fc(features_final)
        output = output.view(output.shape[0],1)

        return output

    def mod_fft(self, x):
        """
        input_shape -> (num_bacth, time, channels)
        """
        eps = 1e-10
        x_mod = torch.abs(self.compute_STFT(x)) # (num_bacth, time, num_mod_freq, 2, num_features)
        x_mod = torch.log(x_mod.pow(2).sum(-2)+eps) # take log of power and combine real and imaginary parts
        x_mod_ave = torch.mean(x_mod,axis=1) # average over time axis -> (num_bacth, num_mod_freq, num_features)
        return x_mod_ave


    def weighted_sum(self, features):
        """
        Returns a weighted sum of outputs from all layers.
        Num_layers -> 1.
        """
        layer_num = len(features)

        # Perform the weighted sum
        stacked_feature = torch.stack(features, dim=0)

        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(layer_num, -1)
        norm_weights = F.softmax(self.weights_stack, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)

        return weighted_feature

#################################


class SSL_diagnoser_mod_alone(nn.Module):
    def __init__(
        self,
        encoder_choice: str = 'wavlm',
        ssl_encoder_source: str = "microsoft/wavlm-base-plus",
        num_ssl_feat: int = 768,
        num_fc_neurons:int=-1,
        num_classes: int = 1,
        freeze_encoder: bool = True,
        pooling_2: str = 'atn',
        sample_rate: int = 50,
        win_length: int = 128,
        hop_length: int = 32,
        dp = 0.25,
        *args,
        **kwargs
        ):
        
        super().__init__(*args, **kwargs)

        self.processor = AutoFeatureExtractor.from_pretrained(ssl_encoder_source)
        self.feature_extractor = WavLMModel.from_pretrained(ssl_encoder_source)

        for param in self.feature_extractor.parameters():
            param.requires_grad = not freeze_encoder

        num_input_feat_2 = num_ssl_feat*2 if pooling_2 == 'atn' else num_ssl_feat
        if num_fc_neurons == -1: num_fc_neurons = num_input_feat_2

        self.fc = nn.Sequential(
            nn.Linear(num_input_feat_2, num_fc_neurons),
            nn.Dropout(p=dp),
            nn.LeakyReLU(0.1),
            nn.Linear(num_fc_neurons, num_classes)
        )

        self.weights_stack = nn.Parameter(torch.ones(self.feature_extractor.config.num_hidden_layers))

        self.compute_STFT = STFT(sample_rate=sample_rate, win_length=win_length, hop_length=hop_length)

        if pooling_2 == "avg":
            self.pooling_2 = lambda x: F.adaptive_avg_pool1d(x, 1)
        elif pooling_2 == 'atn':
            self.pooling_2 = AttentiveStatisticsPooling(
                num_ssl_feat,
                attention_channels=num_ssl_feat,
                global_context=True
            )

    def forward(self, x):
        # Preprocessing the data
        input_values = self.processor(x, sampling_rate=16000, return_tensors="pt").input_values[0]
        input_values = input_values.to(device=x.device, dtype=x.dtype)

        # Extract wav2vec2 hidden states and perform a weighted sum
        features = self.feature_extractor(input_values, output_hidden_states=True)
        features = self.weighted_sum(features.hidden_states[1:]) # (batch, time, features)

        # Calculate modulation energies
        mod_features = self.mod_fft(features)
        mod_features = mod_features.permute(0, 2, 1) # (batch, features, mod_freq)
        mod_features = self.pooling_2(mod_features).squeeze(-1) # (batch, features)

        features_final = mod_features
        output = self.fc(features_final)
        output = output.view(output.shape[0],1)

        return output

    def mod_fft(self, x):
        """
        input_shape -> (num_bacth, time, channels)
        """
        eps = 1e-10
        x_mod = torch.abs(self.compute_STFT(x)) # (num_bacth, time, num_mod_freq, 2, num_features)
        x_mod = x_mod.pow(2).sum(-2) # combine real and imaginary parts
        x_mod_ave = torch.mean(x_mod,axis=1) # average over time axis -> (num_bacth, num_mod_freq, num_features)
        x_mod_ave = torch.log(x_mod_ave+eps) # take log of power
        return x_mod_ave


    def weighted_sum(self, features):
        """
        Returns a weighted sum of outputs from all layers.
        Num_layers -> 1.
        """
        layer_num = len(features)

        # Perform the weighted sum
        stacked_feature = torch.stack(features, dim=0)

        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(layer_num, -1)
        norm_weights = F.softmax(self.weights_stack, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)

        return weighted_feature

#######################


class SSL_vanilla(nn.Module):
    def __init__(
        self,
        encoder_choice: str = 'wavlm',
        ssl_encoder_source: str = "microsoft/wavlm-base-plus",
        pooling: str = "atn",
        num_atn_channels: int = 768, 
        num_input_features: int = 768*2,
        num_fc_neurons:int = -1,
        num_classes: int = 1,
        freeze_encoder: bool = True,
        dp = 0.25,
        *args,
        **kwargs
        ):
        
        super().__init__(*args, **kwargs)

        self.processor = AutoFeatureExtractor.from_pretrained(ssl_encoder_source)
        self.feature_extractor = WavLMModel.from_pretrained(ssl_encoder_source)

        for param in self.feature_extractor.parameters():
            param.requires_grad = not freeze_encoder
        
        if num_fc_neurons == -1: num_fc_neurons = num_atn_channels
        self.fc = nn.Sequential(
            nn.Linear(num_input_features, num_fc_neurons),
            nn.Dropout(p=dp),
            nn.LeakyReLU(0.1),
            nn.Linear(num_fc_neurons, num_classes)
        )

        self.weights_stack = nn.Parameter(torch.ones(self.feature_extractor.config.num_hidden_layers))
        
        if pooling == "avg":
            self.pooling = lambda x: F.adaptive_avg_pool1d(x, 1)
        elif pooling == 'atn':
            self.pooling = AttentiveStatisticsPooling(
                num_atn_channels,
                attention_channels=num_atn_channels,
                global_context=True
            )

    def forward(self, x):
        # Preprocessing the data
        input_values = self.processor(x, sampling_rate=16000, return_tensors="pt").input_values[0]
        input_values = input_values.to(device=x.device, dtype=x.dtype)

        # Extract wav2vec2 hidden states and perform a weighted sum
        features = self.feature_extractor(input_values, output_hidden_states=True)
        features = self.weighted_sum(features.hidden_states[1:])

        # Pooling (on time dimension)
        features = features.permute(0, 2, 1) # (batch, time, features )=> (batch, features, time)
        features = self.pooling(features).squeeze(-1)
        output = self.fc(features)

        return output

    def weighted_sum(self, features):
        """
        Returns a weighted sum of outputs from all layers.
        Num_layers -> 1.
        """
        layer_num = len(features)

        # Perform the weighted sum
        stacked_feature = torch.stack(features, dim=0)

        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(layer_num, -1)
        norm_weights = F.softmax(self.weights_stack, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)

        return weighted_feature