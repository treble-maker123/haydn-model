import torch
from time import time
import torch.nn.functional as F
from pdb import set_trace

def train(models, optims, loader, **kwargs):
    model_names = kwargs.get("model_names",
                    ["forward_", "backward_", "harmony_", "judge_"])
    device = kwargs.get("device", torch.device("cpu"))
    print_iter = kwargs.get("print_iter", 40)
    loss_weights = kwargs.get("loss_weights", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # these are the weights used to balance out the losses
    fp, bp, hp, fr, j, p = loss_weights
    # set all of the models to train mode and correct device
    for key in models:
        models[key].to(device)
        models[key].train()

    stats = {
        "losses": [],
        "fp_losses": [],
        "bp_losses": [],
        "hp_losses": [],
        "fr_losses": [],
        "j_losses": [],
        "p_losses": [],
        "fp_accs": [],
        "bp_accs": [],
        "hp_accs": [],
        "fr_accs": [],
        "j_accs": [],
        "p_accs": []
    }

    start = time()
    for num_iter, X in enumerate(loader):
        X = X.to(device=device)
        batch_size, num_parts, win_len, _ = X.shape
        # index of the note being predicted by forward and backward models
        mid_idx = win_len // 2

        if num_iter % print_iter == 0:
            print("Train iter {}/{}:".format(num_iter, len(loader)))

        for part_idx in range(num_parts):
            # zero out the gradients for all optimizers.
            for key in optims: optims[key].zero_grad()

            try:
                forward_pitch, forward_rhythm, backward_pitch, harmony_pitch, harmony_part, judge_decision \
                    = forward_pass(models, X, part_idx, device=device, model_names=model_names)
            except Exception as error:
                print("Ran into error in iteration {}, skipping, \n{}"
                        .format(num_iter, error))
                continue

            # ===========================
            # back-propagate
            # ===========================
            gt_pitch = X[:,part_idx,mid_idx,0].long()
            gt_rhythm = X[:,part_idx,mid_idx,1].float()
            gt_part = torch.zeros(batch_size).long()
            gt_part[:] = part_idx

            forward_pitch_loss = F.cross_entropy(forward_pitch, gt_pitch)
            backward_pitch_loss = F.cross_entropy(backward_pitch, gt_pitch)
            harmony_pitch_loss = F.cross_entropy(harmony_pitch, gt_pitch)
            forward_rhythm_loss = F.l1_loss(forward_rhythm, gt_rhythm)
            judge_loss = F.cross_entropy(judge_decision, gt_pitch)
            part_loss = F.cross_entropy(harmony_part, gt_part)

            loss = fp * forward_pitch_loss
            loss += bp * backward_pitch_loss
            loss += hp * harmony_pitch_loss
            loss += fr * forward_rhythm_loss
            loss += j * judge_loss
            loss += p * part_loss

            # time to learn stuff
            loss.backward()
            optims[part_idx].step()

            # detach the LSTM hidden states so don't need to retain graph
            forward_model = models[model_names[0] + str(part_idx)]
            backward_model = models[model_names[1] + str(part_idx)]
            fm_h, fm_c = forward_model.hidden
            bm_h, bm_c = backward_model.hidden
            forward_model.hidden = fm_h.detach(), fm_c.detach()
            backward_model.hidden = bm_h.detach(), bm_c.detach()

            # calculate accuracy for each of the models
            with torch.no_grad():
                fp_acc = (forward_pitch.argmax(dim=1) == gt_pitch).sum()*100
                fp_acc /= batch_size
                bp_acc = (backward_pitch.argmax(dim=1) == gt_pitch).sum()*100
                bp_acc /= batch_size
                hp_acc = (harmony_pitch.argmax(dim=1) == gt_pitch).sum()*100
                hp_acc /= batch_size
                fr_acc = ((forward_rhythm > 0.5).float() == gt_rhythm).sum()*100
                fr_acc /= batch_size
                j_acc = (judge_decision.argmax(dim=1) == gt_pitch).sum()*100
                j_acc /= batch_size
                p_acc = (harmony_part.argmax(dim=1) == gt_part).sum()*100
                p_acc /= batch_size

            # save stats for later
            stats['losses'].append(loss.item())
            stats['fp_losses'].append(forward_pitch_loss.item())
            stats['bp_losses'].append(backward_pitch_loss.item())
            stats['hp_losses'].append(harmony_pitch_loss.item())
            stats['fr_losses'].append(forward_rhythm_loss.item())
            stats['j_losses'].append(judge_loss.item())
            stats['p_losses'].append(part_loss.item())
            stats['fp_accs'].append(fp_acc)
            stats['bp_accs'].append(bp_acc)
            stats['hp_accs'].append(hp_acc)
            stats['fr_accs'].append(fr_acc)
            stats['j_accs'].append(j_acc)
            stats['p_accs'].append(p_acc)

            if num_iter % print_iter == 0:
                print(
                    "\tPart {} - ".format(part_idx + 1) +
                    "fp_loss: {:.5f}/{:.2f}%, "
                        .format(forward_pitch_loss.item(), fp_acc) +
                    "bp_loss: {:.5f}/{:.2f}%, "
                        .format(backward_pitch_loss.item(), bp_acc) +
                    "hp_loss: {:.5f}/{:.2f}%, "
                        .format(harmony_pitch_loss.item(), hp_acc) +
                    "j_loss: {:.5f}/{:.2f}%, "
                        .format(judge_loss.item(), j_acc) +
                    "\n\t\tfr_loss: {:.5f}/{:.2f}%, "
                        .format(forward_rhythm_loss.item(), fr_acc) +
                    "p_loss: {:.5f}/{:.2f}%, ".format(part_loss.item(), p_acc) +
                    "\n\t\ttotal weighted loss: {:.5f}".format(loss.item())
                )

        if num_iter % print_iter == 0:
            print("\tTraining time elapsed: {:.2f} seconds"
                    .format(time() - start))
            print("")

    return stats, models

def validate(models, loader, **kwargs):
    model_names = kwargs.get("model_names", None)
    device = kwargs.get("device", torch.device("cpu"))
    print_iter = kwargs.get("print_iter", 40)
    loss_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # these are the weights used to balance out the losses
    fp, bp, hp, fr, j, p = loss_weights
    # set all of the models to train mode and correct device
    for key in models:
        models[key].to(device)
        models[key].eval()

    stats = {
        "losses": [],
        "fp_losses": [],
        "bp_losses": [],
        "hp_losses": [],
        "fr_losses": [],
        "j_losses": [],
        "p_losses": [],
        "fp_accs": [],
        "bp_accs": [],
        "hp_accs": [],
        "fr_accs": [],
        "j_accs": [],
        "p_accs": []
    }

    with torch.no_grad():
        start = time()
        for num_iter, X in enumerate(loader):
            X = X.to(device=device)
            batch_size, num_parts, win_len, _ = X.shape
            # index of the note being predicted by forward and backward models
            mid_idx = win_len // 2

            if num_iter % print_iter == 0:
                print("Valid iter {}/{}:".format(num_iter, len(loader)))

            for part_idx in range(num_parts):
                try:
                    forward_pitch, forward_rhythm, backward_pitch, \
                    harmony_pitch, harmony_part, judge_decision \
                        = forward_pass(models, X, part_idx,
                                       device=device, model_names=model_names)
                except Exception as error:
                    print("Ran into error in iteration {}, skipping"
                            .format(num_iter))
                    continue

                gt_pitch = X[:,part_idx,mid_idx,0].long()
                gt_rhythm = X[:,part_idx,mid_idx,1].float()
                gt_part = torch.zeros(batch_size).long()
                gt_part[:] = part_idx

                forward_pitch_loss = F.cross_entropy(forward_pitch, gt_pitch)
                backward_pitch_loss = F.cross_entropy(backward_pitch, gt_pitch)
                harmony_pitch_loss = F.cross_entropy(harmony_pitch, gt_pitch)
                forward_rhythm_loss = F.l1_loss(forward_rhythm, gt_rhythm)
                judge_loss = F.cross_entropy(judge_decision, gt_pitch)
                part_loss = F.cross_entropy(harmony_part, gt_part)

                loss = fp * forward_pitch_loss
                loss += bp * backward_pitch_loss
                loss += hp * harmony_pitch_loss
                loss += fr * forward_rhythm_loss
                loss += j * judge_loss
                loss += p * part_loss

                # calculate accuracy for each of the models
                fp_acc = (forward_pitch.argmax(dim=1) == gt_pitch).sum()*100
                fp_acc /= batch_size
                bp_acc = (backward_pitch.argmax(dim=1) == gt_pitch).sum()*100
                bp_acc /= batch_size
                hp_acc = (harmony_pitch.argmax(dim=1) == gt_pitch).sum()*100
                hp_acc /= batch_size
                fr_acc = ((forward_rhythm > 0.5).float() == gt_rhythm).sum()*100
                fr_acc /= batch_size
                j_acc = (judge_decision.argmax(dim=1) == gt_pitch).sum()*100
                j_acc /= batch_size
                p_acc = (harmony_part.argmax(dim=1) == gt_part).sum()*100
                p_acc /= batch_size

                # save stats for later
                stats['losses'].append(loss.item())
                stats['fp_losses'].append(forward_pitch_loss.item())
                stats['bp_losses'].append(backward_pitch_loss.item())
                stats['hp_losses'].append(harmony_pitch_loss.item())
                stats['fr_losses'].append(forward_rhythm_loss.item())
                stats['j_losses'].append(judge_loss.item())
                stats['p_losses'].append(part_loss.item())
                stats['fp_accs'].append(fp_acc)
                stats['bp_accs'].append(bp_acc)
                stats['hp_accs'].append(hp_acc)
                stats['fr_accs'].append(fr_acc)
                stats['j_accs'].append(j_acc)
                stats['p_accs'].append(p_acc)

                if num_iter % print_iter == 0:
                    print(
                        "\tPart {} - ".format(part_idx + 1) +
                        "fp_loss: {:.5f}/{:.2f}%, "
                            .format(forward_pitch_loss.item(), fp_acc) +
                        "bp_loss: {:.5f}/{:.2f}%, "
                            .format(backward_pitch_loss.item(), bp_acc) +
                        "hp_loss: {:.5f}/{:.2f}%, "
                            .format(harmony_pitch_loss.item(), hp_acc) +
                        "j_loss: {:.5f}/{:.2f}%, "
                            .format(judge_loss.item(), j_acc) +
                        "\n\t\tfr_loss: {:.5f}/{:.2f}%, "
                            .format(forward_rhythm_loss.item(), fr_acc) +
                        "p_loss: {:.5f}/{:.2f}%, "
                            .format(part_loss.item(), p_acc) +
                        "\n\t\ttotal weighted loss: {:.5f}".format(loss.item())
                    )

            if num_iter % print_iter == 0:
                print("\t`Validation time elapsed: {:.2f} seconds"
                    .format(time() - start))
                print("`")

    return stats, models

def forward_pass(models, data, part_idx, **kwargs):
    '''
    Forward pass with one of the parts.
    '''
    device = kwargs.get("device", torch.device("cpu"))
    model_names = kwargs.get("model_names",
                    ["forward_", "backward_", "harmony_", "judge_"])

    batch_size, num_parts, win_len, _ = data.shape

    embed_model = models['pitch_embed']
    forward_model = models[model_names[0] + str(part_idx)]
    backward_model = models[model_names[1] + str(part_idx)]
    harmony_model = models[model_names[2] + str(part_idx)]
    judge_model = models[model_names[3] + str(part_idx)]

    # index of the note being predicted by forward and backward models
    mid_idx = win_len // 2

    part = data[:, part_idx, :, :]
    pitches = part[:, :, 0].long()
    rhythms = part[:, :, 1].float()

    # ===========================
    # forward and backward models
    # ===========================
    # embed the pitches and turn it into a part
    embedded_pitches = embed_model(pitches)
    embedded_part = torch \
                        .cat((embedded_pitches,rhythms[:,:,None]),dim=2)
    # swap axis 1 and 0 so it's [seq_len, batch_size..] for LSTM
    embedded_part = embedded_part.permute(1,0,2)
    # forward model
    forward_pitch, forward_rhythm = \
        forward_model(embedded_part[:mid_idx,:,:])
    # backward model
    backward_pitch, _ = backward_model(embedded_part[mid_idx+1:,:,:])

    # ===========================
    # harmony model
    # ===========================
    # mark which part the note is coming from, which will help guide
    # the network.
    part_ind = torch.zeros(batch_size, num_parts, num_parts)
    for i in range(num_parts): part_ind[:,i,i] = 1
    # embed the four pitches in the middle
    pitches = data[:, :, mid_idx, 0].long()
    embedded_pitches = embed_model(pitches)
    harmony_input = torch.cat((embedded_pitches, part_ind), dim=2)
    # zero out the part that is being guessed
    harmony_input[:,part_idx,:] = 0
    # output shape is [BATCH_SIZE, VOCAB_SIZE + 4]
    harmony_output = harmony_model(harmony_input)
    harmony_pitch = harmony_output[:, :-4]
    harmony_part = harmony_output[:, -4:]

    # ===========================
    # judge model
    # ===========================
    choices = torch.cat((forward_pitch[:,None,:],
                            backward_pitch[:,None,:],
                            harmony_pitch[:,None,:]), dim=1).argmax(dim=2)
    embedded_choies = embed_model(choices)
    judge_decision = judge_model(embedded_choies)

    return forward_pitch, forward_rhythm.squeeze(dim=1), \
           backward_pitch, harmony_pitch, \
           harmony_part, judge_decision
