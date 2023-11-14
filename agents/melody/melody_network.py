import torch
import torch.nn as nn

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


class Melody_Network(nn.Module):
    def __init__(self):
        super(Melody_Network, self).__init__()
        self.device = DEVICE

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=4)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=289,
            hidden_size=256,
            dropout=0.3,
            num_layers=6,
            bidirectional=True,
            batch_first=True,
        )

        self.dense_x1 = nn.Linear(in_features=306, out_features=129)

        self.dense_pitch1 = nn.Linear(in_features=129, out_features=129)
        self.dense_duration1 = nn.Linear(in_features=129, out_features=16)
        self.dense_pitch2 = nn.Linear(in_features=129, out_features=129)
        self.dense_duration2 = nn.Linear(in_features=129, out_features=16)

        self.dense_upsample = nn.Linear(in_features=288, out_features=291)

        self.last_dense_pitch = nn.Linear(in_features=289, out_features=129)
        self.last_dense_duration = nn.Linear(in_features=289, out_features=16)

        self.downscale_for_lstm = nn.Linear(in_features=512, out_features=286)

        self.lstm_dense1 = nn.Linear(in_features=512, out_features=286)
        self.lstm_dense2 = nn.Linear(in_features=512, out_features=286)

        self.initialize_weights()

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

    def predictive_network(
        self, inputs_conv, inputs_lstm, accumulated_time, time_left_on_chord
    ):
        inputs_conv_pooled = torch.mean(inputs_conv, dim=1)

        # upsampled_cov = self.upsample_FC(inputs_conv_pooled)

        combined = inputs_conv_pooled + inputs_lstm

        concated = torch.cat((combined, accumulated_time, time_left_on_chord), dim=1)

        x_dense1 = self.dense_x1(concated)

        pitch_output = self.dense_pitch(x_dense1)
        duration_output = self.dense_duration(x_dense1)

        return pitch_output, duration_output

    def predictive_network1(
        self, inputs_conv, inputs_lstm, accumulated_time, time_left_on_chord
    ):
        inputs_conv_pooled = torch.mean(inputs_conv, dim=1)

        upsampled_cov = inputs_conv_pooled

        combined = inputs_conv_pooled + inputs_lstm

        concated = torch.cat((combined, accumulated_time, time_left_on_chord), dim=1)

        x_dense1 = self.dense_x1(concated)

        pitch_output = self.dense_pitch1(x_dense1)
        duration_output = self.dense_duration1(x_dense1)

        return pitch_output, duration_output

    def predictive_network2(
        self, inputs_conv, inputs_lstm, accumulated_time, time_left_on_chord
    ):
        inputs_conv_pooled = torch.mean(inputs_conv, dim=1)

        # upsampled_cov = self.upsample_FC(inputs_conv_pooled)

        combined = inputs_conv_pooled + inputs_lstm

        concated = torch.cat((combined, accumulated_time, time_left_on_chord), dim=1)

        x_dense1 = self.dense_x1(concated)

        pitch_output = self.dense_pitch2(x_dense1)
        duration_output = self.dense_duration2(x_dense1)

        return pitch_output, duration_output

    def upsample_FC(self, inputs):
        x = self.dense_upsample(inputs)
        return x

    def conv_block(self, inputs):
        # Pass through the Conv1d and ReLU

        x = self.conv1d(inputs.float())
        x = self.relu(x)

        return x

    def lstm_block(self, inputs):
        lstm_output, _ = self.lstm(inputs)
        x1 = self.lstm_dense1(lstm_output)
        x2 = self.lstm_dense2(lstm_output)
        return x1, x2

    def last_layer(
        self,
        pitch_input1,
        pitch_input2,
        duration_input1,
        duration_input2,
        current_chord,
        next_chord,
    ):
        pitch_output = pitch_input1 + pitch_input2
        duration_output = duration_input1 + duration_input2

        x = torch.cat((pitch_output, duration_output, current_chord, next_chord), dim=1)

        x_pitch = self.last_dense_pitch(x)
        x_duration = self.last_dense_duration(x)

        return x_pitch, x_duration

    def forward(self, inputs, accumulated_time, time_left_on_chord):
        inputs = (
            inputs.clone()
            .detach()
            .to(device=self.device, dtype=self.conv1d.weight.dtype)
        )
        pitches_and_duration = inputs[:, :145]

        current_chord = inputs[:, 145:217]
        next_chord = inputs[:, 217:289]

        x = inputs.unsqueeze(1)  # Add channel dimension, [batch, 1, features]

        x_conv = self.conv_block(x)
        first_conv_output = x_conv[:, :145, :]
        second_conv_output = x_conv[:, 145:, :]

        # The output of Conv1d is [batch, channels, new_steps], we want to get rid of 'channels' for LSTM input
        # x = inputs.permute(0, 2, 1)  # Change to [batch, steps, features]

        x_lstm1, x_lstm2 = self.lstm_block(inputs)

        # pitch_output, duration_output = self.predictive_network(
        #     x_conv, x_lstm, accumulated_time, time_left_on_chord
        # )

        pitch_output1, duration_output1 = self.predictive_network1(
            first_conv_output, x_lstm1, accumulated_time, time_left_on_chord
        )
        pitch_output2, duration_output2 = self.predictive_network2(
            second_conv_output, x_lstm2, accumulated_time, time_left_on_chord
        )

        pitch, duration = self.last_layer(
            pitch_output1,
            pitch_output2,
            duration_output1,
            duration_output2,
            current_chord,
            next_chord,
        )

        return pitch, duration


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
