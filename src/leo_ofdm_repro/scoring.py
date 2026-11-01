"""Committed campaign behavior; source mapping in _config/extraction.json."""
import math
class InvalidEvidence(ValueError):pass
Hold=InvalidEvidence
FS=30.72e6
DELAY_LIMIT_SAMPLES=FS/18.36e6

def score_emissions(events, *, present, truth_at, terminal_s):
    if not math.isfinite(terminal_s) or terminal_s < 0:
        raise Hold('Invalid terminal availability')
    checked = []
    seen = set()
    for (index, event) in enumerate(events):
        key = event.get('event_index', index)
        if key in seen:
            raise Hold('Duplicate emitted event')
        seen.add(key)
        available = float(event['available_s'])
        if not math.isfinite(available) or not 0 <= available <= terminal_s + 1e-06:
            raise Hold('Event outside its terminal interval')
        horizon = float(event['observation_s'])
        if available + 1e-06 < horizon:
            raise Hold('Output predates required samples')
        prediction = event['prediction']
        correct = False
        error = None
        if present and event['alert']:
            truth = truth_at(horizon)
            error = [float(prediction[1]) - truth[0], float(prediction[2]) - truth[1]]
            correct = abs(error[0]) <= DELAY_LIMIT_SAMPLES and abs(error[1]) <= 1.0
        checked.append({**event, 'correct': correct, 'errors_samples_Hz': error})
    checked.sort(key=lambda event: event['available_s'])
    positives = [event for event in checked if event['alert']]
    correct = [event for event in positives if event['correct']]
    first_correct = correct[0] if correct else None
    final = checked[-1] if checked else None
    return {'present': bool(present), 'events': checked, 'first_correct_available_s': first_correct['available_s'] if first_correct else None, 'miss': bool(present and (not correct)), 'wrong_first': bool(present and positives and (not positives[0]['correct'])), 'any_false_alert': bool(not present and positives), 'final_correct': bool(final and final['correct']), 'final_false_alert': bool(not present and final and final['alert']), 'terminal_s': terminal_s, 'operational_s': first_correct['available_s'] if first_correct else terminal_s}

def events_from_attempt(attempt):
    if 'updates' in attempt:
        return [dict(update, event_index=index) for (index, update) in enumerate(attempt['updates'])]
    return [{'event_index': 0, 'observation_s': attempt['observation_s'], 'available_s': attempt['decision_s'], 'alert': attempt['output']['alert'], 'prediction': attempt['output']['prediction']}]

def mixture_time(target_mean, empty_mean, target_weight=0.5):
    if not 0 <= target_weight <= 1:
        raise Hold('Invalid declared mixture weight')
    return target_weight * target_mean + (1 - target_weight) * empty_mean

def standardized_metrics(rows, *, target_weight=0.5):
    targets = [row for row in rows if row['present']]
    empties = [row for row in rows if not row['present']]
    if not targets or not empties:
        raise Hold('Both strata are needed for the declared mixture')
    mean = lambda values: sum(values) / len(values)
    target_time = mean([row['operational_s'] for row in targets])
    empty_time = mean([row['terminal_s'] for row in empties])
    detection = mean([not row['miss'] for row in targets])
    false_alert = mean([row['any_false_alert'] for row in empties])
    return {'target_attempts': len(targets), 'empty_attempts': len(empties), 'correct_detection_probability': detection, 'empty_any_false_alert_probability': false_alert, 'target_operational_mean_s': target_time, 'empty_terminal_mean_s': empty_time, 'target_weight': target_weight, 'standardized_operational_mean_s': mixture_time(target_time, empty_time, target_weight), 'standardized_correct_decision_probability': mixture_time(detection, 1 - false_alert, target_weight)}
