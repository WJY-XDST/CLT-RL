"""Small frozen-policy interventions for diagnosis, never policy acceptance."""
import math
import re


def command_probe_cases(specification, command_config):
    """Use in-distribution commands to diagnose training-only failures."""
    if not isinstance(specification, list) or not 1 <= len(specification) <= 50:
        raise ValueError('Command diagnosis needs one to fifty cases')
    bounds = dict(speed=(command_config.ranges_reverse_lin_vel_x[0], command_config.ranges_lin_vel_x[1]),
                  yaw=command_config.ranges_ang_vel_yaw, height=command_config.ranges_height)
    names, result = set(), []
    for case in specification:
        if set(case) != {'name', 'speed', 'yaw', 'height'} or not re.fullmatch(r'[a-z][a-z0-9_]*', case['name']):
            raise ValueError('Invalid diagnostic command case')
        if case['name'] in names:
            raise ValueError('Duplicate diagnostic command name')
        names.add(case['name'])
        for key, (low, high) in bounds.items():
            value = case[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError('Diagnostic command outside training range: ' + key)
        result.append(tuple(case[key] for key in ('name', 'speed', 'yaw', 'height')))
    return result


def command_probe_plan(specification, command_config):
    if isinstance(specification, list):
        return command_probe_cases(specification, command_config), None, None
    if not isinstance(specification, dict) or set(specification) != {'phase_seconds', 'phases'}:
        raise ValueError('Expected fixed command cases or a grouped-command diagnostic sequence')
    period = specification['phase_seconds']
    if period != command_config.resampling_time:
        raise ValueError('Diagnostic command period must match the training sampler')
    phases = specification['phases']
    if not 2 <= len(phases) <= 10:
        raise ValueError('Command diagnosis needs two to ten phases')
    phases = [command_probe_cases(phase, command_config) for phase in phases]
    if any([case[0] for case in phase] != [case[0] for case in phases[0]] for phase in phases):
        raise ValueError('Every command phase must retain the same case identities')
    return phases[0], [[case[1:] for case in phase] for phase in phases], float(period)


def build_bias_cases(cases, specification):
    """Expand selected commands with paired, bounded action offsets."""
    if set(specification) != {'cases', 'probes'}:
        raise ValueError('Bias diagnosis accepts only cases and probes')
    selected, probes = specification['cases'], specification['probes']
    known = {case[0] for case in cases}
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= known:
        raise ValueError('Unknown or duplicate diagnostic command case')
    if not 2 <= len(probes) <= 25:
        raise ValueError('Use one unchanged control and at most 24 bounded interventions')
    names = set()
    for probe in probes:
        if set(probe) != {'name', 'bias'} or not re.fullmatch(r'[a-z][a-z0-9_]*', probe['name']):
            raise ValueError('Invalid diagnostic probe specification')
        if probe['name'] in names:
            raise ValueError('Duplicate diagnostic probe name')
        names.add(probe['name'])
        bias = probe['bias']
        if len(bias) != 6 or any(not isinstance(x, (int, float)) or not math.isfinite(x) or abs(x) > .05 for x in bias):
            raise ValueError('Bias must contain six finite normalized offsets bounded by .05')
    if probes[0]['name'] != 'unchanged' or any(probes[0]['bias']):
        raise ValueError('First probe must retain the unchanged frozen policy')
    expanded, biases, labels = [], [], []
    for case in cases:
        if case[0] not in selected:
            continue
        for probe in probes:
            expanded.append((case[0] + '__' + probe['name'], *case[1:]))
            biases.append(probe['bias'])
            labels.append(dict(command_case=case[0], probe=probe['name'], normalized_bias=probe['bias']))
    return expanded, biases, labels


def add_action_bias(means, biases):
    """Leave the recorded actor means untouched; environment limits still apply."""
    if means.shape != biases.shape or means.shape[-1] != 6:
        raise ValueError('One six-dimensional bias is required per environment')
    return means + biases


def wheel_reference_probe(applied, raw, limit):
    """Isolate a software reference clamp with frozen weights and motor gains."""
    if not math.isfinite(limit) or not 1. <= limit <= 2.:
        raise ValueError('Diagnostic wheel action limit must be between one and two')
    if applied.shape != raw.shape or raw.ndim != 2 or raw.shape[-1] != 6:
        raise ValueError('Expected paired six-dimensional actions')
    result = applied.clone()
    result[:, (2, 5)] = raw[:, (2, 5)].clamp(-limit, limit)
    return result
