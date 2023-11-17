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

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class Melody_Network(nn.Module):
    class Tier3LSTM(nn.Module):
        def __init__(self):
            super(Melody_Network.Tier3LSTM, self).__init__()
            self.lstm = nn.LSTM(
                input_size=289,
                hidden_size=256,
                dropout=0.3,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
            )

            self.downscale = nn.Linear(in_features=512, out_features=289)

        def forward(self, input_sequence):
            x = self.lstm(input_sequence)[0][:, -1, :]
            x = self.downscale(x)
            return x

    class Tier2LSTM(nn.Module):
        def __init__(self):
            super(Melody_Network.Tier2LSTM, self).__init__()
            self.lstm = nn.LSTM(
                input_size=289,
                hidden_size=256,
                dropout=0.3,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
            )

            self.downscale = nn.Linear(in_features=512, out_features=289)

        def forward(self, inputs_sequence, tier3_output, tier2_output):
            if tier2_output is None:
                combined = inputs_sequence + tier3_output.unsqueeze(1)
            else:
                combined = (
                    inputs_sequence
                    + tier3_output.unsqueeze(1)
                    + tier2_output.unsqueeze(1)
                )
            x = self.lstm(combined)[0][:, -1, :]
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
                dropout=0.3,
                bidirectional=True,
                batch_first=True,
            )

        def forward(self, input):
            x = self.conv1d(input)
            x = self.relu(x)
            x = x.permute(0, 2, 1)
            x = self.lstm(x)[0][:, -1, :]
            return x

    class PredictiveNetwork(nn.Module):
        def __init__(self):
            super(Melody_Network.PredictiveNetwork, self).__init__()
            self.upscale = nn.Linear(in_features=289, out_features=512)

            self.FC_pitch = nn.Linear(in_features=516, out_features=129)
            self.FC_duration = nn.Linear(in_features=516, out_features=16)

            self.FC_pitch_full = nn.Linear(in_features=661, out_features=129)
            self.FC_duration_full = nn.Linear(in_features=661, out_features=16)

        def forward(
            self,
            inputs_conv,
            inputs_lstm_tier2,
            inputs_lstm_tier3,
            accumulated_time,
            time_left_on_chord,
            previous_pitch,
            previous_duration,
        ):
            # inputs_conv = F.adaptive_avg_pool1d(inputs_conv, 1).squeeze(2)

            inputs_lstm_tier2 = self.upscale(inputs_lstm_tier2)
            inputs_lstm_tier3 = self.upscale(inputs_lstm_tier3)

            combined = inputs_conv + inputs_lstm_tier2 + inputs_lstm_tier3

            # concat = torch.cat((combined, accumulated_time, time_left_on_chord), dim=1)
            if previous_pitch == None:
                concat = torch.cat(
                    (combined, accumulated_time),
                    dim=1,
                )

                pitch_output = self.FC_pitch(concat)
                duration_output = self.FC_duration(concat)
                return pitch_output, duration_output

            concat = torch.cat(
                (combined, accumulated_time, previous_pitch, previous_duration), dim=1
            )

            pitch_output = self.FC_pitch_full(concat)
            duration_output = self.FC_duration_full(concat)

            return pitch_output, duration_output

    def __init__(self):
        super(Melody_Network, self).__init__()
        self.device = DEVICE
        self._create_tier2_lstms()
        self._create_tier3_lstms()
        self._create_predictive_networks()
        self._create_conv_networks()

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

        tier_3_outputs.append(self.tier3_lstm[0](event_1_to_8))
        tier_3_outputs.append(self.tier3_lstm[1](event_9_to_16))

        # Pass data through tier 2
        tier_2_outputs = []
        for idx, cell in enumerate(self.tier2_lstms):
            if idx < 4:
                if idx == 0:
                    tier_2_outputs.append(cell(events[idx], tier_3_outputs[0], None))
                tier_2_outputs.append(
                    cell(events[idx], tier_3_outputs[0], tier_2_outputs[idx - 1])
                )
            else:
                tier_2_outputs.append(
                    cell(events[idx], tier_3_outputs[0], tier_2_outputs[idx - 1])
                )

        # Pass data through tier 1
        conv_outputs = []
        for idx, cell in enumerate(self.conv_networks):
            conv_outputs.append(cell.forward(conv_events[idx]))

        # Pass data through predictive network
        predictive_outputs = []
        for idx, cell in enumerate(self.predictive_networks):
            if idx == 0:
                predictive_outputs.append(
                    cell.forward(
                        conv_outputs[math.floor(idx / 4)],
                        tier_2_outputs[math.floor(idx / 2) + 1],
                        tier_3_outputs[math.floor(idx / 8)],
                        accumulated_time_chunk[idx],
                        time_left_on_chord_chunk[idx],
                        None,
                        None,
                    )
                )
            else:
                predictive_outputs.append(
                    cell.forward(
                        conv_outputs[math.floor(idx / 4)],
                        tier_2_outputs[math.floor(idx / 2) + 1],
                        tier_3_outputs[math.floor(idx / 8)],
                        accumulated_time_chunk[idx],
                        time_left_on_chord_chunk[idx],
                        predictive_outputs[idx - 1][0],
                        predictive_outputs[idx - 1][1],
                    )
                )

        output_vector = [[], []]
        for i in range(len(predictive_outputs)):
            output_vector[0].append(predictive_outputs[i][0]),
            output_vector[1].append(predictive_outputs[i][1]),

        output_vector = [torch.stack(output_vector[0]), torch.stack(output_vector[1])]
        output_vector[0] = output_vector[0].permute(1, 0, 2)
        output_vector[1] = output_vector[1].permute(1, 0, 2)

        return output_vector


"""class Melody_Network(nn.Module):
    def __init__(self):
        super(Melody_Network, self).__init__()
        self.hidden_size = HIDDEN_SIZE_MELODY
        self.num_layers = NUM_LAYERS_MELODY

        # Embedding layers
        self.pitch_embedding = nn.Embedding(PITCH_SIZE_MELODY, HIDDEN_SIZE_MELODY)
        self.duration_embedding = nn.Embedding(DURATION_SIZE_MELODY, HIDDEN_SIZE_MELODY)
        self.current_chord_embedding = nn.Embedding(
            CHORD_SIZE_MELODY, HIDDEN_SIZE_MELODY
        )
        self.next_chord_embedding = nn.Embedding(CHORD_SIZE_MELODY, HIDDEN_SIZE_MELODY)
        self.bar_embedding = nn.Embedding(
            2, HIDDEN_SIZE_MELODY
        )  # Only two options for start/end of bar

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=HIDDEN_SIZE_MELODY * 5,  # Combined embedding size
            hidden_size=HIDDEN_SIZE_MELODY,
            num_layers=NUM_LAYERS_MELODY,
            batch_first=True,
            bidirectional=True,
        )

        # Output layers
        self.pitch_out = nn.Linear(
            HIDDEN_SIZE_MELODY * 2, PITCH_SIZE_MELODY
        )  # Times 2 for bidirectional
        self.duration_out = nn.Linear(HIDDEN_SIZE_MELODY * 2, DURATION_SIZE_MELODY)

    def forward(self, pitch, duration, current_chord, next_chord, bars):
        # Check the size of inputs
        print("Pitch size:", pitch.size())
        print("Duration size:", duration.size())
        print("Chords size:", current_chord.size())
        print("Chords size:", next_chord.size())
        print("Bars size:", bars.size())

        # Embed each input
        pitch_embedded = self.pitch_embedding(pitch)
        duration_embedded = self.duration_embedding(duration)
        current_chord_embedded = self.current_chord_embedding(current_chord)
        next_chord_embedded = self.next_chord_embedding(next_chord)

        bar_embedded = self.bar_embedding(bars)

        # Ensure that all embeddings are of the correct size
        print("Pitch embedded size:", pitch_embedded.size())
        print("Duration embedded size:", duration_embedded.size())
        print("Chord embedded size:", current_chord_embedded.size())
        print("Chord embedded size:", next_chord_embedded.size())
        print("Bar embedded size:", bar_embedded.size())

        # Concatenate the embeddings
        x = torch.cat(
            (
                pitch_embedded,
                duration_embedded,
                current_chord_embedded,
                next_chord_embedded,
                bar_embedded,
            ),
            1,
        )
        print(x.shape)
        print(x)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Pass the output of LSTM to the output layers
        pitch_out = self.pitch_out(lstm_out)
        duration_out = self.duration_out(lstm_out)

        return pitch_out, duration_out"""

'''class Melody_Network(object):
    def __init__(self):
        self.batch_size = BATCH_SIZE
        self.big_frame_size = BIG_FRAME_SIZE
        self.frame_size = FRAME_SIZE

        self.rnn_type = RNN_TYPE
        self.dim = DIM
        self.birnndim = BIRNNDIM
        self.piano_dim = PIANO_DIM
        self.n_rnn = N_RNN
        self.seq_len = SEQ_LEN
        self.mode_choice = MODE_CHOICE
        self.note_channel = NOTE_CHANNEL
        self.rhythm_channel = RHYTHM_CHANNEL
        self.chord_channel = CHORD_CHANNEL
        self.bar_channel = BAR_CHANNEL
        self.alpha1 = ALPHA1
        self.alpha2 = ALPHA2
        self.drop_out_keep_prob = DROP_OUT
        if IF_COND == "cond":
            self.if_cond = True
            print("model setting: conditional")
        elif IF_COND == "no_cond":
            self.if_cond = False
            print("model setting: unconditional")

        def single_cell(if_attention=False, atten_len=None):
            if self.rnn_type == "GRU":
                return tf.keras.layers.GRU(
                    self.dim,
                    return_sequences=True,
                    return_state=True,
                    dropout=1 - self.drop_out_keep_prob,
                )
            elif self.rnn_type == "LSTM":
                return tf.keras.layers.LSTM(
                    self.dim,
                    return_sequences=True,
                    return_state=True,
                    dropout=1 - self.drop_out_keep_prob,
                )

        def single_birnncell(self):
            if self.rnn_type == "GRU":
                return tf.keras.layers.GRU(
                    self.birnndim,
                    return_sequences=True,
                    return_state=True,
                    dropout=1 - self.drop_out_keep_prob,
                )

            elif self.rnn_type == "LSTM":
                return tf.keras.layers.LSTM(
                    self.birnndim,
                    return_sequences=True,
                    return_state=True,
                    dropout=1 - self.drop_out_keep_prob,
                )

        if self.n_rnn > 1:
            attn_cell_lst = [
                single_cell(if_attention=True, atten_len=2 * self.frame_size)
            ]
            attn_cell_lst += [single_cell() for _ in range(self.n_rnn - 1)]
            self.attn_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(attn_cell_lst)
            # self.attn_cell = tf.keras.layers.StackedRNNCells(attn_cell_lst)
            self.sample_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [single_cell() for _ in range(self.n_rnn)]
            )
            self.frame_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [single_cell() for _ in range(self.n_rnn)]
            )
            self.big_frame_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [single_cell() for _ in range(self.n_rnn)]
            )
            self.birnn_fwcell = single_birnncell(self)
            self.birnn_bwcell = single_birnncell(self)
        else:
            self.attn_cell = single_cell(
                if_attention=True, atten_len=2 * self.frame_size
            )
            self.sample_cell = single_cell()
            self.frame_cell = single_cell()
            self.big_frame_cell = single_cell()
            self.birnn_fwcell = single_birnncell(self)
            self.birnn_bwcell = single_birnncell(self)

    def weight_bias(self, tensor_in, dim, name):
        with tf.compat.v1.variable_scope(name):
            W_initializer = tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1.0, mode="fan_avg", distribution="uniform"
            )
            b_initializer = tf.compat.v1.constant_initializer()
            W = tf.compat.v1.get_variable(
                name=name + "_W",
                shape=(tensor_in.shape[-1], dim),
                initializer=W_initializer,
                dtype=tf.float32,
                trainable=True,
            )
            b = tf.compat.v1.get_variable(
                name=name + "_b",
                shape=(dim),
                initializer=b_initializer,
                dtype=tf.float32,
                trainable=True,
            )
            out = tf.add(tf.matmul(tensor_in, W), b)
        return out

    def birnn(self, all_chords):
        outputs_tuple, _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(
            cell_fw=self.birnn_fwcell,
            cell_bw=self.birnn_bwcell,
            inputs=all_chords,
            dtype=tf.float32,
        )

        # outputs_tuple,_ = tf.nn.bidirectional_dynamic_rnn(cell_fw = forward_cell, cell_bw = backward_cell,inputs = all_chords, dtype=tf.float32)
        outputs = tf.concat([outputs_tuple[0], outputs_tuple[1]], axis=-1)
        # all_chords_birnn = self.weight_bias(outputs, 2*self.chord_channel ,"birnn_weights_3")
        all_chords_birnn = self.weight_bias(
            outputs, self.chord_channel, "birnn_weights_3"
        )
        return all_chords_birnn

    def big_frame_level(self, big_frame_input, big_frame_state=None):
        big_frame_input_chunks = tf.reshape(
            big_frame_input,
            [
                -1,  # batch
                int(big_frame_input.shape[1]) // self.big_frame_size,  # no_of_chunks
                self.big_frame_size * int(big_frame_input.shape[-1]),
            ],
        )  # frame_size*merged_dim
        with tf.compat.v1.variable_scope("BIG_FRAME_RNN"):
            if big_frame_state is not None:  # during generation
                (
                    big_frame_outputs_all_stps,
                    big_frame_last_state,
                ) = tf.compat.v1.nn.dynamic_rnn(
                    self.big_frame_cell,
                    big_frame_input_chunks,
                    initial_state=big_frame_state,
                    dtype=tf.float32,
                )

            else:  # during training
                (
                    big_frame_outputs_all_stps,
                    big_frame_last_state,
                ) = tf.compat.v1.nn.dynamic_rnn(
                    self.big_frame_cell, big_frame_input_chunks, dtype=tf.float32
                )  # batch, no_chunks, dim

            big_frame_outputs_all_upsample = self.weight_bias(
                big_frame_outputs_all_stps,
                self.dim * self.big_frame_size // self.frame_size,
                "upsample",
            )  # batch, no_chunks, dim*big_size/small_size
            big_frame_outputs = tf.reshape(
                big_frame_outputs_all_upsample,
                [
                    tf.shape(big_frame_outputs_all_upsample)[0],
                    tf.shape(big_frame_outputs_all_upsample)[1]
                    * self.big_frame_size
                    // self.frame_size,
                    self.dim,
                ],
            )  # (batch, no_frame_chunks*ratio, dim)

            return big_frame_outputs, big_frame_last_state

    def frame_level_switch(
        self, frame_input, frame_state=None, bigframe_output=None, if_rs=True
    ):
        frame_input_chunks = tf.reshape(
            frame_input,
            [
                -1,  # batch
                int(frame_input.shape[1]) // self.frame_size,  # no_of_chunks
                self.frame_size * int(frame_input.shape[-1]),
            ],
        )  # frame_size*merged_dim
        with tf.compat.v1.variable_scope("FRAME_RNN"):
            if bigframe_output is not None:
                frame_input_chunks = self.weight_bias(
                    frame_input_chunks, self.dim, "emb_frame_chunks"
                )
                frame_input_chunks += bigframe_output  # batch, no_chunk, dim

            if frame_state is not None:  # during generation
                frame_outputs_all_stps, frame_last_state = tf.compat.v1.nn.dynamic_rnn(
                    self.frame_cell,
                    frame_input_chunks,
                    initial_state=frame_state,
                    dtype=tf.float32,
                )
            else:  # during training
                frame_outputs_all_stps, frame_last_state = tf.compat.v1.nn.dynamic_rnn(
                    self.frame_cell, frame_input_chunks, dtype=tf.float32
                )
            if bigframe_output is not None and if_rs is True:  # residual connection
                frame_outputs_all_stps += (
                    bigframe_output  # batch, no_chunk, dim + batch, no_chunk, dim
                )
            frame_outputs_all_upsample = self.weight_bias(
                frame_outputs_all_stps, self.dim * self.frame_size, "upsample2"
            )
            frame_outputs = tf.reshape(
                frame_outputs_all_upsample,
                [
                    tf.shape(frame_outputs_all_upsample)[0],
                    tf.shape(frame_outputs_all_upsample)[1] * self.frame_size,
                    self.dim,
                ],
            )  # (batch, n_frame*frame_size, dim)
            return frame_outputs, frame_last_state

    def sample_level(self, sample_input_sequences, frame_output=None, rm_time=None):
        sample_filter_shape = [
            self.frame_size,
            sample_input_sequences.shape[-1],
            self.dim,
        ]
        sample_filter = tf.compat.v1.get_variable(
            "sample_filter",
            sample_filter_shape,
            initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1.0, mode="fan_avg", distribution="uniform"
            ),
        )
        mlp_out = tf.nn.conv1d(
            input=sample_input_sequences,
            filters=sample_filter,
            stride=1,
            padding="VALID",
            name="sample_conv",
        )  # (batch, seqlen-framesize, dim)
        if frame_output is not None:
            logits = mlp_out + frame_output
        if rm_time is not None:
            logits = tf.concat([logits, rm_time], axis=-1)

        rhythm_logits = self.weight_bias(logits, self.rhythm_channel, "rhythm_weights1")
        rhythm_logits = tf.nn.relu(rhythm_logits)
        rhythm_logits = self.weight_bias(
            rhythm_logits, self.rhythm_channel, "rhythm_weights2"
        )

        bar_logits = self.weight_bias(rhythm_logits, self.bar_channel, "bar_weights")

        note_logits = self.weight_bias(logits, self.note_channel, "note_weights1")
        note_logits = tf.nn.relu(note_logits)
        note_logits = self.weight_bias(note_logits, self.note_channel, "note_weights2")

        return (
            rhythm_logits,
            bar_logits,
            note_logits,
        )  # return (batch, pred_length, piano_dim)

    def bln_attn(self, baseline_input, baseline_state=None, if_attn=True):
        if if_attn:
            print("cell choice is attn")
            cell = self.attn_cell
        else:
            cell = self.sample_cell

        with tf.compat.v1.variable_scope("baseline"):
            if baseline_state is None:  # during training
                bln_outputs_all, baseline_last_state = tf.compat.v1.nn.dynamic_rnn(
                    cell, baseline_input, dtype=tf.float32
                )  # batch, no_chunks, dim

            else:  # during generation
                bln_outputs_all, baseline_last_state = tf.compat.v1.nn.dynamic_rnn(
                    cell, baseline_input, initial_state=baseline_state, dtype=tf.float32
                )

            """baseline_outputs_all_stps = self.weight_bias(bln_outputs_all, self.piano_dim-self.chord_channel ,"dense_weights_bln")
        return baseline_outputs_all_stps, baseline_last_state"""

        rhythm_logits = self.weight_bias(
            bln_outputs_all, self.rhythm_channel, "rhythm_weights1"
        )
        rhythm_logits = tf.nn.relu(rhythm_logits)
        rhythm_logits = self.weight_bias(
            rhythm_logits, self.rhythm_channel, "rhythm_weights2"
        )

        bar_logits = self.weight_bias(rhythm_logits, self.bar_channel, "bar_weights")

        note_logits = self.weight_bias(
            bln_outputs_all, self.note_channel, "note_weights1"
        )
        note_logits = tf.nn.relu(note_logits)
        note_logits = self.weight_bias(note_logits, self.note_channel, "note_weights2")

        return (
            rhythm_logits,
            bar_logits,
            note_logits,
            baseline_last_state,
        )  # return (batch, pred_length, piano_dim)

    def _create_network_2t_fc_tweek_last_layer(self, one_t_input):
        print("####MODEL:BAR...####")
        sample_input = one_t_input[:, :-1, :]

        frame_input = one_t_input[:, : -self.frame_size, :]

        ##frame_level##
        frame_outputs, final_frame_state = self.frame_level_switch(frame_input)
        ##sample_level##
        rhythm_logits, bar_logits, note_logits = self.sample_level(
            sample_input, frame_output=frame_outputs
        )

        return rhythm_logits, bar_logits, note_logits

    def _create_network_3t_fc_tweek_last_layer(self, two_t_input, if_rs=False):
        print("3t_fc")
        # big frame level
        big_frame_input = two_t_input[:, : -self.big_frame_size, :]

        big_frame_outputs, final_big_frame_state = self.big_frame_level(big_frame_input)

        # frame level
        frame_input = two_t_input[
            :, self.big_frame_size - self.frame_size : -self.frame_size, :
        ]

        frame_outputs, final_frame_state = self.frame_level_switch(
            frame_input, bigframe_output=big_frame_outputs, if_rs=if_rs
        )

        ##sample level
        sample_input = two_t_input[:, self.big_frame_size - self.frame_size : -1, :]

        rhythm_logits, bar_logits, note_logits = self.sample_level(
            sample_input, frame_output=frame_outputs
        )

        return rhythm_logits, bar_logits, note_logits

    def _create_network_ad_rm2t_fc_tweek_last_layer(self, one_t_input, rm_tm):
        sample_input = one_t_input[:, :-1, :]  # batch, seq-1, piano_dim

        frame_input = one_t_input[
            :, : -self.frame_size, :
        ]  # (batch, seq-frame_size, piano_dim)
        remaining_time_input = rm_tm  # (batch, seq-frame_size, piano_dim)
        ##frame_level##
        frame_outputs, final_frame_state = self.frame_level_switch(frame_input)
        ##sample_level##
        rhythm_logits, bar_logits, note_logits = self.sample_level(
            sample_input, frame_output=frame_outputs, rm_time=remaining_time_input
        )
        return rhythm_logits, bar_logits, note_logits

    def _create_network_ad_rm3t_fc_tweek_last_layer(
        self, two_t_input, rm_tm=None, if_rs=True
    ):
        print("_create_network_ad_rm3t_fc")
        with tf.compat.v1.name_scope("CMHRNN_net"):
            sample_input = two_t_input[:, self.big_frame_size - self.frame_size : -1, :]

            frame_input = two_t_input[
                :, self.big_frame_size - self.frame_size : -self.frame_size, :
            ]

            big_frame_input = two_t_input[:, : -self.big_frame_size, :]

            big_frame_outputs, final_big_frame_state = self.big_frame_level(
                big_frame_input
            )

            frame_outputs, final_frame_state = self.frame_level_switch(
                frame_input, bigframe_output=big_frame_outputs, if_rs=if_rs
            )

            remaining_time_input = rm_tm  # (batch, seq-frame_size, piano_dim)
            ##sample_level##
            rhythm_logits, bar_logits, note_logits = self.sample_level(
                sample_input, frame_output=frame_outputs, rm_time=remaining_time_input
            )

            return rhythm_logits, bar_logits, note_logits

    def _create_network_bln_attn_fc(self, baseline_input, if_attn=False):
        rhythm_logits, bar_logits, note_logits, _ = self.bln_attn(
            baseline_input, if_attn=if_attn
        )
        return rhythm_logits, bar_logits, note_logits

    def loss_CMHRNN(
        self, X, y, rm_time=None, l2_regularization_strength=None, name="sample"
    ):
        self.X = X
        self.y = y
        self.rm_time = rm_time

        with tf.compat.v1.name_scope(name):
            if self.mode_choice == "ad_rm2t_fc":  # 2 tier with acc time
                (
                    pd_sustain,
                    pd_bar,
                    pd_note,
                ) = self._create_network_ad_rm2t_fc_tweek_last_layer(
                    one_t_input=self.X, rm_tm=self.rm_time
                )  # (batch* seq_len-frame, self.note + self.rhythm)
                pd = tf.concat([pd_bar, pd_sustain, pd_note], axis=-1)
            elif self.mode_choice == "ad_rm3t_fc":  # 3 tier with acc time
                (
                    pd_sustain,
                    pd_bar,
                    pd_note,
                ) = self._create_network_ad_rm3t_fc_tweek_last_layer(
                    two_t_input=self.X, rm_tm=self.rm_time, if_rs=False
                )  # (batch* seq_len-frame, self.note + self.rhythm)
                pd = tf.concat([pd_bar, pd_sustain, pd_note], axis=-1)
            elif (
                self.mode_choice == "ad_rm3t_fc_rs"
            ):  # 3 tier with acc time with residual conn.
                (
                    pd_sustain,
                    pd_bar,
                    pd_note,
                ) = self._create_network_ad_rm3t_fc_tweek_last_layer(
                    two_t_input=self.X, rm_tm=self.rm_time, if_rs=True
                )  # (batch* seq_len-frame, self.note + self.rhythm)
                pd = tf.concat([pd_bar, pd_sustain, pd_note], axis=-1)
            elif self.mode_choice == "bln_attn_fc":  # attention baseline
                pd_sustain, pd_bar, pd_note = self._create_network_bln_attn_fc(
                    baseline_input=self.X, if_attn=True
                )  # (batch* seq_len-frame, self.note + self.rhythm)
                pd = tf.concat([pd_bar, pd_sustain, pd_note], axis=-1)
            elif self.mode_choice == "bln_fc":  # vanilla rnn
                pd_sustain, pd_bar, pd_note = self._create_network_bln_attn_fc(
                    baseline_input=self.X, if_attn=False
                )  # (batch* seq_len-frame, self.note + self.rhythm)
                pd = tf.concat([pd_bar, pd_sustain, pd_note], axis=-1)
            elif self.mode_choice == "2t_fc":  # 2 tier no acc time
                (
                    pd_sustain,
                    pd_bar,
                    pd_note,
                ) = self._create_network_2t_fc_tweek_last_layer(
                    one_t_input=self.X
                )  # (batch* seq_len-frame, self.note + self.rhythm)
                pd = tf.concat([pd_bar, pd_sustain, pd_note], axis=-1)
            elif self.mode_choice == "3t_fc":  # 3 tier no acc time
                (
                    pd_sustain,
                    pd_bar,
                    pd_note,
                ) = self._create_network_3t_fc_tweek_last_layer(
                    two_t_input=self.X, if_rs=True
                )  # (batch* seq_len-frame, self.note + self.rhythm)
                pd = tf.concat([pd_bar, pd_sustain, pd_note], axis=-1)

            gt_bar = self.y[:, :, : self.bar_channel]
            gt_bar = tf.reshape(gt_bar, [-1, self.bar_channel])
            gt_sustain = self.y[
                :, :, self.bar_channel : self.bar_channel + self.rhythm_channel
            ]
            gt_sustain = tf.reshape(gt_sustain, [-1, self.rhythm_channel])
            gt_note = self.y[:, :, self.bar_channel + self.rhythm_channel :]
            gt_note = tf.reshape(gt_note, [-1, self.note_channel])

            pd_sustain = tf.reshape(pd_sustain, [-1, self.rhythm_channel])
            pd_bar = tf.reshape(pd_bar, [-1, self.bar_channel])
            pd_note = tf.reshape(pd_note, [-1, self.note_channel])

            with tf.compat.v1.name_scope("sample_RNN_loss"):
                loss_note = tf.nn.softmax_cross_entropy_with_logits(
                    logits=pd_note, labels=tf.stop_gradient(gt_note)
                )
                loss_rhythm = tf.nn.softmax_cross_entropy_with_logits(
                    logits=pd_sustain, labels=tf.stop_gradient(gt_sustain)
                )
                loss_bar = tf.nn.softmax_cross_entropy_with_logits(
                    logits=pd_bar, labels=tf.stop_gradient(gt_bar)
                )
                loss = (
                    self.alpha1 * loss_note
                    + self.alpha2 * loss_rhythm
                    + (1 - self.alpha1 - self.alpha2) * loss_bar
                )

                reduced_loss = tf.reduce_mean(loss)
                reduced_note_loss = tf.reduce_mean(loss_note)
                reduced_rhythm_loss = tf.reduce_mean(loss_rhythm)
                reduced_bar_loss = tf.reduce_mean(loss_bar)
                tf.compat.v1.summary.scalar("loss", reduced_loss)
                tf.compat.v1.summary.scalar("note_loss", reduced_note_loss)
                tf.compat.v1.summary.scalar("rhythm_loss", reduced_rhythm_loss)
                tf.compat.v1.summary.scalar("bar_loss", reduced_bar_loss)
                if l2_regularization_strength is None:
                    return self.y, pd, reduced_loss
                else:
                    l2_loss = tf.add_n(
                        [
                            tf.nn.l2_loss(v)
                            for v in tf.compat.v1.trainable_variables()
                            if not ("bias" in v.name)
                        ]
                    )
                    total_loss = reduced_loss + l2_regularization_strength * l2_loss
                    tf.compat.v1.summary.scalar("l2_loss", l2_loss)
                    tf.compat.v1.summary.scalar("total_loss", total_loss)
                    return self.y, pd, total_loss
'''
