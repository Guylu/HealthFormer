"""Visit boundaries for artifacts that encode calendar-year groups."""

def identify_visits(modalities, times, marker, year_index):
    """Return (first_end, second_start, second_end, second_year), or an exclusion reason.

    End positions are exclusive. These are year groups, not clinical visit IDs.
    Marker repetitions within a year do not create new groups. Reject unordered
    records rather than including intervening measurements in a future target.
    """
    years = times[:, year_index]
    if years.numel() < 2:
        return None, "too_short"
    if bool((years[1:] < years[:-1]).any()):
        return None, "unordered_years"
    marker_years = years[modalities == marker].unique(sorted=True)
    if marker_years.numel() < 2:
        return None, "fewer_than_two_marker_years"
    first, second = marker_years[:2]
    first_end = int((years == first).nonzero()[-1].item()) + 1
    second_positions = (years == second).nonzero().flatten()
    second_start = int(second_positions[0].item())
    second_end = int(second_positions[-1].item()) + 1
    return (first_end, second_start, second_end, int(second.item())), None

ELIS_TRAIN_SUFFIXES = (
    "_BloodTestsLoader", "_ItamarSleepLoader", "_BodyMeasuresLoader", "_ECGTextLoader",
    "_ABILoader", "_RetinaScanLoader", "_MicroStats", "_UltrasoundLoader", "_DEXALoader",
    "_questionires", "medication", "medical_condition_code",
)
ELIS_EVAL_SUFFIXES = ELIS_TRAIN_SUFFIXES + ("_NightingaleLoader", "_locations")


def elis_modalities(name_to_number, *, evaluation=False):
    suffixes = ELIS_EVAL_SUFFIXES if evaluation else ELIS_TRAIN_SUFFIXES
    return {number for name, number in name_to_number.items()
            if any(suffix in name for suffix in suffixes)}
