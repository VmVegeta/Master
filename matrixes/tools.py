from measure import Measure


def get_difference_in_hours(measure: Measure, result: Measure):
    measure_time = measure.datetime
    result_time = result.datetime
    diff = result_time - measure_time
    seconds = diff.total_seconds()
    return int(seconds / 60 / 60)
