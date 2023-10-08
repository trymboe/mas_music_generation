import os
import math
import time
import torch
import itertools

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from .utils import create_exp_dir, create_dir_if_not_exists

from config import (
    LEARNING_RATE_DRUM,
    MAX_STEP_DRUM,
    ETA_MIN_DRUM,
    WORK_DIR,
    TRAIN_BATCH_SIZE_DRUM,
    NUM_TOKENS_PREDICT_DRUM,
    EXTENDED_CONTEXT_LENGTH_DRUM,
    EVAL_BATCH_SIZE_DRUM,
    N_ALL_PARAMS,
    N_NONEMB_PARAMS,
    MEM_LEN,
    MAX_EVAL_STEPS_DRUM,
    VARLEN,
    CLIP_DRUM,
    WARMUP_STEPS_DRUM,
    LOG_INTERVAL_DRUM,
    EVAL_INTERVAL_DRUM,
    DEBUG,
)


def train_drum(model: nn.Module, dataset: Dataset, device: torch.device):
    logging = create_exp_dir(WORK_DIR, scripts_to_save=None, debug=WORK_DIR)

    tr_iter = dataset.get_iterator(
        "train",
        TRAIN_BATCH_SIZE_DRUM,
        NUM_TOKENS_PREDICT_DRUM,
        device=device,
        ext_len=EXTENDED_CONTEXT_LENGTH_DRUM,
    )
    va_iter = dataset.get_iterator(
        "valid",
        EVAL_BATCH_SIZE_DRUM,
        NUM_TOKENS_PREDICT_DRUM,
        device=device,
        ext_len=EXTENDED_CONTEXT_LENGTH_DRUM,
    )
    te_iter = dataset.get_iterator(
        "test",
        EVAL_BATCH_SIZE_DRUM,
        NUM_TOKENS_PREDICT_DRUM,
        device=device,
        ext_len=EXTENDED_CONTEXT_LENGTH_DRUM,
    )

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_DRUM)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, MAX_STEP_DRUM, eta_min=ETA_MIN_DRUM
    )
    para_model = model.to(device)

    def train():
        nonlocal train_step, train_loss, best_val_loss, eval_start_time, log_start_time
        # Turn on training mode which enables dropout.
        model.train()

        mems = tuple()
        train_iter = tr_iter.get_varlen_iter() if VARLEN else tr_iter

        for batch, (data, target, seq_len) in enumerate(train_iter):
            model.zero_grad()

            ret = para_model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.float().mean().type_as(loss)
            loss.backward()
            train_loss += loss.float().item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_DRUM)

            optimizer.step()

            # step-wise learning rate annealing
            train_step += 1

            # linear warmup stage
            if train_step < WARMUP_STEPS_DRUM:
                curr_lr = LEARNING_RATE_DRUM * train_step / WARMUP_STEPS_DRUM
                optimizer.param_groups[0]["lr"] = curr_lr
            else:
                scheduler.step()

            # Logging
            if train_step % LOG_INTERVAL_DRUM == 0:
                cur_loss = train_loss / LOG_INTERVAL_DRUM
                elapsed = time.time() - log_start_time
                log_str = (
                    "| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} "
                    "| ms/batch {:5.2f} | loss {:5.2f}".format(
                        epoch,
                        train_step,
                        batch + 1,
                        optimizer.param_groups[0]["lr"],
                        elapsed * 1000 / LOG_INTERVAL_DRUM,
                        cur_loss,
                    )
                )
                log_str += " | ppl {:9.3f}".format(math.exp(cur_loss))
                logging(log_str)
                train_loss = 0
                log_start_time = time.time()

            if train_step == 1 or train_step % EVAL_INTERVAL_DRUM == 0:
                val_loss = evaluate(va_iter)
                logging("-" * 100)
                log_str = (
                    "| Eval {:3d} at step {:>8d} | time: {:5.2f}s "
                    "| valid loss {:5.2f}".format(
                        train_step // EVAL_INTERVAL_DRUM,
                        train_step,
                        (time.time() - eval_start_time),
                        val_loss,
                    )
                )
                log_str += " | valid ppl {:9.3f}".format(math.exp(val_loss))
                logging(log_str)
                logging("-" * 100)
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    create_dir_if_not_exists(
                        os.path.join(WORK_DIR, f"train_step_{train_step}", "")
                    )
                    if not DEBUG:
                        with open(
                            os.path.join(
                                WORK_DIR,
                                f"train_step_{train_step}",
                                "model.pt",
                            ),
                            "wb",
                        ) as f:
                            torch.save(model, f)
                        with open(
                            os.path.join(
                                WORK_DIR,
                                f"train_step_{train_step}",
                                "optimizer.pt",
                            ),
                            "wb",
                        ) as f:
                            torch.save(optimizer.state_dict(), f)
                    best_val_loss = val_loss

                eval_start_time = time.time()

            if train_step == MAX_STEP_DRUM:
                with open(
                    os.path.join(
                        WORK_DIR,
                        "drum_model.pt",
                    ),
                    "wb",
                ) as f:
                    torch.save(model, f)
                with open(
                    os.path.join(
                        WORK_DIR,
                        "optimizer.pt",
                    ),
                    "wb",
                ) as f:
                    torch.save(optimizer.state_dict(), f)
                break

    def evaluate(eval_iter):
        # Turn on evaluation mode which disables dropout.
        model.eval()

        # If the model does not use memory at all, make the ext_len longer.
        # Otherwise, make the mem_len longer and keep the ext_len the same.
        if MEM_LEN == 0:
            model.reset_length(
                NUM_TOKENS_PREDICT_DRUM,
                EXTENDED_CONTEXT_LENGTH_DRUM
                + NUM_TOKENS_PREDICT_DRUM
                - NUM_TOKENS_PREDICT_DRUM,
                MEM_LEN,
            )
        else:
            model.reset_length(
                NUM_TOKENS_PREDICT_DRUM,
                EXTENDED_CONTEXT_LENGTH_DRUM,
                MEM_LEN + NUM_TOKENS_PREDICT_DRUM - NUM_TOKENS_PREDICT_DRUM,
            )

        # Evaluation
        total_len, total_loss = 0, 0.0
        with torch.no_grad():
            mems = tuple()
            for i, (data, target, seq_len) in enumerate(eval_iter):
                if MAX_EVAL_STEPS_DRUM > 0 and i >= MAX_EVAL_STEPS_DRUM:
                    break
                ret = model(data, target, *mems)
                loss, mems = ret[0], ret[1:]
                loss = loss.mean()
                total_loss += seq_len * loss.float().item()
                total_len += seq_len

        # Switch back to the training mode
        model.reset_length(
            NUM_TOKENS_PREDICT_DRUM, EXTENDED_CONTEXT_LENGTH_DRUM, MEM_LEN
        )
        model.train()

        return total_loss / total_len

    # TRAINING STARTS HERE
    train_step = 0
    train_loss = 0
    best_val_loss = None

    log_start_time = time.time()
    eval_start_time = time.time()

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in itertools.count(start=1):
            train()
            if train_step == MAX_STEP_DRUM:
                logging("-" * 100)
                logging("End of training")
                break
    except KeyboardInterrupt:
        logging("-" * 100)
        logging("Exiting from training early")

    # Create dir if not exist
    create_dir_if_not_exists(WORK_DIR)

    # Load the best saved model.
    with open(os.path.join(WORK_DIR, "drum_model.pt"), "rb") as f:
        model = torch.load(f)
    para_model = model.to(device)

    # Run on test data.
    test_loss = evaluate(te_iter)
    logging("=" * 100)
    logging(
        "| End of training | test loss {:5.2f} | test ppl {:9.3f}".format(
            test_loss, math.exp(test_loss)
        )
    )
    logging("=" * 100)
