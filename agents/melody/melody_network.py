import torch
import torch.nn as nn
import math

import tensorflow as tf


from config import (
    PITCH_SIZE_MELODY,
    DURATION_SIZE_MELODY,
    CHORD_SIZE_MELODY,
    HIDDEN_SIZE_MELODY,
    NUM_LAYERS_MELODY,
    DEVICE,
)

CONCAT = True

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class Melody_Network(nn.Module):
    class Tier3LSTM(nn.Module):
        def __init__(self):
            super(Melody_Network.Tier3LSTM, self).__init__()
            self.lstm = nn.LSTM(
                input_size=197,
                hidden_size=256,
                dropout=0.3,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
            )
            self.device = DEVICE
            self.downscale = nn.Linear(in_features=512, out_features=197)

        def forward(self, input_sequence, previous_cell_output=None):
            if previous_cell_output is not None:
                if CONCAT:
                    new_input = torch.cat(
                        (input_sequence, previous_cell_output.unsqueeze(1)), dim=1
                    )
                else:
                    new_input = input_sequence + previous_cell_output.unsqueeze(1)
            else:
                new_input = input_sequence
            new_input.to(self.device)
            x = self.lstm(new_input)[0][:, -1, :]
            x = self.downscale(x)
            return x

    class Tier2LSTM(nn.Module):
        def __init__(self):
            super(Melody_Network.Tier2LSTM, self).__init__()
            self.lstm = nn.LSTM(
                input_size=197,
                hidden_size=256,
                dropout=0.3,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
            )
            self.device = DEVICE
            self.downscale = nn.Linear(in_features=512, out_features=197)

        def forward(self, inputs_sequence, tier3_output, tier2_output=None):
            if tier2_output is not None:
                if CONCAT:
                    new_input = torch.cat(
                        (
                            inputs_sequence,
                            tier3_output.unsqueeze(1),
                            tier2_output.unsqueeze(1),
                        ),
                        dim=1,
                    )
                else:
                    new_input = (
                        inputs_sequence
                        + tier3_output.unsqueeze(1)
                        + tier2_output.unsqueeze(1)
                    )
            else:
                if CONCAT:
                    new_input = torch.cat(
                        (inputs_sequence, tier3_output.unsqueeze(1)), dim=1
                    )
                else:
                    new_input = inputs_sequence + tier3_output.unsqueeze(1)
            new_input.to(self.device)
            x = self.lstm(new_input)[0][:, -1, :]
            x = self.downscale(x)
            return x

    class ConvNetwork(nn.Module):
        def __init__(self):
            super(Melody_Network.ConvNetwork, self).__init__()
            self.conv1d = nn.Conv1d(in_channels=4, out_channels=256, kernel_size=4)
            self.relu = nn.ReLU()
            self.lstm = nn.LSTM(
                input_size=256,
                hidden_size=256,
                num_layers=2,
                dropout=0.3,
                bidirectional=True,
                batch_first=True,
            )
            self.device = DEVICE

        def forward(self, input):
            input.to(self.device)
            x = self.conv1d(input)
            x = self.relu(x)
            x = x.permute(0, 2, 1)
            x = self.lstm(x)[0][:, -1, :]
            return x

    class PredictiveNetwork(nn.Module):
        def __init__(self):
            self.device = DEVICE
            
            in_features2 = 197 + 4 if not CONCAT else 197 * 4 + 4
            in_features1 = 197 + 4 if not CONCAT else 197 * 3 + 4

            super(Melody_Network.PredictiveNetwork, self).__init__()
            self.conv1d = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=4)
            self.relu = nn.ReLU()
            self.downsample_conv = nn.Linear(in_features=256, out_features=197)

            self.FC1 = nn.Linear(in_features=in_features1, out_features=197)
            self.FC2 = nn.Linear(in_features=in_features2, out_features=197)

        def forward(
            self,
            inputs_conv,
            inputs_lstm_tier2,
            inputs_lstm_tier3,
            accumulated_time,
            time_left_on_chord,
            previous_pitch_tier1=None,
        ):
            # inputs_conv = F.adaptive_avg_pool1d(inputs_conv, 1).squeeze(2)

            # print(inputs_conv.shape)
            # print(inputs_lstm_tier2.shape)
            # print(inputs_lstm_tier3.shape)
            # print(accumulated_time.shape)
            # print(time_left_on_chord.shape)
            # print(previous_pitch_tier1.shape)

            inputs_conv.to(self.device)
            x_conv = self.conv1d(inputs_conv.unsqueeze(1))
            x_conv = self.relu(x_conv)
            x_conv = torch.mean(x_conv, dim=2)  # Shape: [64, 256]
            x_conv = self.downsample_conv(x_conv)  # Shape: [64, 197]

            if CONCAT:
                if previous_pitch_tier1 is not None:
                    new_input = torch.cat(
                        (
                            x_conv,
                            inputs_lstm_tier2,
                            inputs_lstm_tier3,
                            previous_pitch_tier1,
                        ),
                        dim=1,
                    )
                else:
                    new_input = torch.cat(
                        (x_conv, inputs_lstm_tier2, inputs_lstm_tier3), dim=1
                    )
            else:
                if previous_pitch_tier1 is not None:
                    new_input = (
                        x_conv
                        + inputs_lstm_tier2
                        + inputs_lstm_tier3
                        + previous_pitch_tier1
                    )
                else:
                    new_input = x_conv + inputs_lstm_tier2 + inputs_lstm_tier3

            
            new_input.to(self.device)

            
            new_input = torch.cat((new_input, accumulated_time), dim=1)
            if not CONCAT:
                x = self.FC(new_input)
                return x
            else:
                if previous_pitch_tier1 is None:
                    x = self.FC1(new_input)
                else:
                    x = self.FC2(new_input)
                return x

    def __init__(self):
        super(Melody_Network, self).__init__()
        self.device = DEVICE
        self._create_tier2_lstms()
        self._create_tier3_lstms()
        self._create_predictive_networks()
        self._create_conv_networks()

        self.FC_pitch = nn.Linear(in_features=197, out_features=PITCH_SIZE_MELODY)
        self.FC_duration = nn.Linear(in_features=197, out_features=DURATION_SIZE_MELODY)

    def _create_conv_networks(self):
        self.conv_networks = nn.ModuleList()
        for i in range(4):
            self.conv_networks.append(self.ConvNetwork())

    def _create_predictive_networks(self):
        self.predictive_networks = nn.ModuleList()
        for i in range(16):
            self.predictive_networks.append(self.PredictiveNetwork())

    def _create_tier2_lstms(self):
        self.tier2_lstms = nn.ModuleList()
        for i in range(8):
            t2_lstm: list[self.Tier2LSTM] = self.Tier2LSTM()
            self.tier2_lstms.append(t2_lstm)

    def _create_tier3_lstms(self):
        self.tier3_lstm = nn.ModuleList()
        for i in range(2):
            t3_lstm: list[self.Tier3LSTM] = self.Tier3LSTM()
            self.tier3_lstm.append(t3_lstm)

    def initialize_weights(self):
        # Initialize weights using Xavier initialization
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                init.xavier_uniform_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # Optional: initializing forget gate bias to 1 or a small value
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)

    def _get_paired_events(self, inputs):
        events = []
        for i in range(0, 16, 2):
            events.append(inputs[:, i : i + 2, :])
        return events

    def _get_conv_events(self, inputs):
        paired_events = []
        for i in range(0, 15, 4):
            paired_events.append(inputs[:, i : i + 4, :])
        return paired_events

    def forward(self, inputs, accumulated_time, time_left_on_chord):
        inputs = (
            inputs.clone()
            .detach()
            .to(device=self.device, dtype=self.conv_networks[0].conv1d.weight.dtype)
        )
        accumulated_time.to(self.device)
        time_left_on_chord.to(self.device)
        
        event_1_to_8 = inputs[:, :8, :]
        event_9_to_16 = inputs[:, -8:, :]

        accumulated_time_chunk = torch.chunk(accumulated_time, 16, dim=1)
        accumulated_time_chunk = [chunk.squeeze(1) for chunk in accumulated_time_chunk]

        time_left_on_chord_chunk = torch.chunk(time_left_on_chord, 16, dim=1)
        time_left_on_chord_chunk = [
            chunk.squeeze(1) for chunk in time_left_on_chord_chunk
        ]

        events = self._get_paired_events(inputs)
        conv_events = self._get_conv_events(inputs)

        # Pass data through tier 3
        tier_3_outputs = []
        for idx, cell in enumerate(self.tier3_lstm):
            if idx == 0:
                tier_3_outputs.append(cell(event_1_to_8))
            else:
                tier_3_outputs.append(cell(event_9_to_16, tier_3_outputs[0]))

        # Pass data through tier 2
        tier_2_outputs = []
        for idx, cell in enumerate(self.tier2_lstms):
            if idx < 4:
                if idx == 0:
                    tier_2_outputs.append(cell(events[idx], tier_3_outputs[0]))
                tier_2_outputs.append(
                    cell(events[idx], tier_3_outputs[0], tier_2_outputs[idx - 1])
                )
            else:
                tier_2_outputs.append(
                    cell(events[idx], tier_3_outputs[1], tier_2_outputs[idx - 1])
                )

        # Pass data through tier 1

        # conv_outputs = []
        # for idx, cell in enumerate(self.conv_networks):
        #     conv_outputs.append(cell.forward(conv_events[idx]))

        # Pass data through predictive network
        predictive_outputs = []
        for idx, cell in enumerate(self.predictive_networks):
            if idx == 0:
                predictive_outputs.append(
                    cell.forward(
                        inputs[:, idx, :],
                        tier_2_outputs[math.floor(idx / 2)],
                        tier_3_outputs[math.floor(idx / 8)],
                        accumulated_time_chunk[idx].to(self.device),
                        time_left_on_chord_chunk[idx],
                    )
                )
            else:
                predictive_outputs.append(
                    cell.forward(
                        inputs[:, idx, :],
                        tier_2_outputs[math.floor(idx / 2)],
                        tier_3_outputs[math.floor(idx / 8)],
                        accumulated_time_chunk[idx].to(self.device),
                        time_left_on_chord_chunk[idx],
                        predictive_outputs[idx - 1],
                    )
                )

        # Pass data through final FC layer
        x_pitch = self.FC_pitch(predictive_outputs[-1])
        x_duration = self.FC_duration(predictive_outputs[-1])

        return x_pitch, x_duration
