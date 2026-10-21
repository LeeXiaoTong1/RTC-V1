"""Receiver frame repair and reliable output guidance, without extra encoding."""
import torch
from torch.nn.utils.rnn import pad_sequence
from w2v_v3183r2.common import GROUPS
from .repair import aligned_reference, source_feature_loss, source_kd_loss, deterministic_head
from .schedule import auxiliary_coefficients


def paired_losses(model, micro, layouts, sequences, processed, main_indices, clean_indices, quotas, cfg):
    zero = sequences[0].float().sum()*0.
    feature, kd = zero, zero
    records = []
    coeff_f, coeff_d = auxiliary_coefficients(cfg)
    if not coeff_f or micro[0][0]['meta_side'] != 'A':
        return feature, kd, records
    selected = [(unit[0], offset, slots) for unit, (offset, slots) in zip(micro, layouts) if unit[0]['role'] == 'noisy']
    if not selected:
        return feature, kd, records
    if not bool(model.repair.scale_ready):
        raise RuntimeError('Calibrate and freeze Clean-RTC frame scales before paired training')
    raw_noisy = [sequences[offset] for _, offset, _ in selected]
    lengths = [len(x) for x in raw_noisy]
    # Detachment separates LF from CE: LF cannot pull the front end or final
    # fused classifier representation toward a possibly incomplete reference.
    prediction = model.repair.auxiliary_prediction(pad_sequence([x.float() for x in raw_noisy], batch_first=True), lengths)
    targets, masks, masses = [], [], []
    for row, offset, slots in selected:
        info = row.get('repair_pair')
        if info is None:
            raise ValueError('Missing CTC-independent receiver pair metadata')
        c, n = sequences[offset+slots['clean']], sequences[offset]
        cv = torch.as_tensor(info['clean_valid'], device=c.device)
        nv = torch.as_tensor(info['noisy_valid'], device=n.device)
        if cv.shape != (len(c),) or nv.shape != (len(n),):
            raise ValueError('Pair mask/Omni frame geometry mismatch')
        target, valid = aligned_reference(c.float(), cv, len(n), info['shift_frames'])
        targets.append(target)
        masks.append(valid & nv)
        group = GROUPS.index((row['language'], row['label']))
        masses.append(.25/quotas[group])
        records.append(dict(source=row['source_id'], origin=row.get('v5_origin_split', 'train'),
            group=f'{row["language"]}/{row["label"]}', delay=info['delay'], frames=len(n)))
    mask = pad_sequence(masks, batch_first=True)
    values, active = source_feature_loss(prediction, pad_sequence(targets, batch_first=True), mask,
        model.repair.frame_scale, cfg['repair_min_frames'])
    with torch.no_grad():
        raw_values, _ = source_feature_loss(pad_sequence([x.detach().float() for x in raw_noisy], batch_first=True),
            pad_sequence(targets, batch_first=True), mask, model.repair.frame_scale, cfg['repair_min_frames'])
    mass = values.new_tensor(masses)
    # The outer A/B reducer supplies 1/2. Compensate here exactly once.
    feature = 2*coeff_f*(values*mass).sum()
    kd_values, kd_active = torch.zeros_like(values), torch.zeros_like(active)
    if coeff_d:
        position = {index: i for i, index in enumerate(main_indices+clean_indices)}
        no = [processed[position[offset]] for _, offset, _ in selected]
        cl = [processed[position[offset+slots['clean']]].detach() for _, offset, slots in selected]
        with deterministic_head(model.head):
            with torch.no_grad():
                teacher = model.classify_processed(cl)
            student = model.classify_processed(no)
        labels = torch.tensor([row['label'] for row, _, _ in selected], device=student.device)
        kd_values, kd_active = source_kd_loss(student, teacher, labels, active,
            cfg['repair_temperature'], cfg['repair_teacher_confidence'], cfg['repair_teacher_advantage'])
        kd = 2*coeff_d*(kd_values*mass).sum()
    # One small transfer per physical microbatch; retain no activation graphs.
    details = torch.stack((values.detach(), raw_values, kd_values.detach(), active.float(), kd_active.float(), mask.sum(1).float()), -1).cpu().tolist()
    for record, (lf, raw, ld, valid, gated, frames) in zip(records, details):
        record.update(feature_loss=lf, raw_feature_loss=raw, kd_loss=ld, valid=bool(valid), kd_active=bool(gated), valid_frames=int(frames))
    return feature, kd, records
