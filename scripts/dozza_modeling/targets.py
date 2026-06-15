"""Target e regole di raggruppamento feature per le analisi Dozza."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


FLOW_TARGETS = ["ingressi_borgo", "uscite_borgo"]
NATIONALITY_TARGETS = ["tim_Ni_mean15", "tim_Ns_mean15"]
AGE_TARGETS = [f"tim_F{idx}_mean15" for idx in range(1, 7)]

CALENDAR_FEATURE_COLUMNS = {
    "year",
    "month",
    "day",
    "hour",
    "dayofweek",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
}

FLOW_LEAKAGE_COLUMNS = {
    "ingressi_borgo",
    "uscite_borgo",
    "entra_all_cameras",
    "uscita_all_cameras",
    "saldo_ingressi_uscite",
    "target_ingressi_complete",
    "target_uscite_complete",
    "target_complete",
}
FLOW_COMPONENT_PREFIXES = ("entra_", "uscita_", "obs_")

FEATURE_SCOPE_CHOICES = [
    "auto",
    "calendar_meteo_tim",
    "calendar_meteo_pedoni",
    "calendar_meteo_tim_pedoni",
    "calendar_meteo",
]


@dataclass(frozen=True)
class TargetSet:
    name: str
    title: str
    targets: list[str]
    default_feature_scope: str
    description: str


TARGET_SETS = {
    "flow": TargetSet(
        name="flow",
        title="Flussi pedonali",
        targets=FLOW_TARGETS,
        default_feature_scope="calendar_meteo_tim_pedoni",
        description="Prevede ingressi e uscite del borgo dai dati TIM, meteo, eventi, varchi pedonali e calendario.",
    ),
    "nationality": TargetSet(
        name="nationality",
        title="Italiani e stranieri",
        targets=NATIONALITY_TARGETS,
        default_feature_scope="calendar_meteo_tim_pedoni",
        description=(
            "Prevede le presenze TIM medie orarie di italiani e stranieri usando "
            "varchi pedonali, TIM, meteo, eventi e calendario."
        ),
    ),
    "age": TargetSet(
        name="age",
        title="Fasce eta'",
        targets=AGE_TARGETS,
        default_feature_scope="calendar_meteo_tim_pedoni",
        description=(
            "Prevede le presenze TIM medie orarie per le sei fasce d'eta' usando "
            "varchi pedonali, TIM, meteo, eventi e calendario."
        ),
    ),
}

TARGET_SET_ALIASES = {
    "flows": "flow",
    "ingressi_uscite": "flow",
    "pedoni": "flow",
    "dual": "nationality",
    "nazionalita": "nationality",
    "italiani_stranieri": "nationality",
    "six": "age",
    "eta": "age",
    "fasce_eta": "age",
}


def target_set_choices() -> list[str]:
    return sorted([*TARGET_SETS.keys(), *TARGET_SET_ALIASES.keys()])


def resolve_target_set(name: str) -> TargetSet:
    canonical = TARGET_SET_ALIASES.get(name, name)
    if canonical not in TARGET_SETS:
        raise ValueError(f"Target set non riconosciuto: {name}")
    return TARGET_SETS[canonical]


def resolved_feature_scope(target_set: TargetSet, feature_scope: str) -> str:
    if feature_scope == "auto":
        return target_set.default_feature_scope
    if feature_scope not in FEATURE_SCOPE_CHOICES:
        raise ValueError(f"Feature scope non riconosciuto: {feature_scope}")
    return feature_scope


def strip_history_suffix(feature: str) -> str:
    """Restituisce la colonna sorgente prima dei suffissi lag/rolling."""

    base = re.sub(r"_lag\d+h$", "", feature)
    base = re.sub(r"_roll\d+h_mean$", "", base)
    return base


def is_history_feature(feature: str) -> bool:
    return bool(re.search(r"_lag\d+h$", feature) or re.search(r"_roll\d+h_mean$", feature))


def tim_family(column: str) -> str | None:
    base = strip_history_suffix(column)
    match = re.match(r"^(tim_[A-Za-z0-9]+)_", base)
    return match.group(1) if match else None


def is_calendar_feature(feature: str) -> bool:
    return feature in CALENDAR_FEATURE_COLUMNS


def is_meteo_feature(feature: str) -> bool:
    return strip_history_suffix(feature).startswith("meteo_")


def is_tim_feature(feature: str) -> bool:
    return strip_history_suffix(feature).startswith("tim_")


def is_pedoni_feature(feature: str) -> bool:
    base = strip_history_suffix(feature)
    if base in {"ingressi_borgo", "uscite_borgo", "saldo_ingressi_uscite"}:
        return True
    if base in {"entra_all_cameras", "uscita_all_cameras"}:
        return True
    return base.startswith("entra_") or base.startswith("uscita_")


def is_event_feature(feature: str) -> bool:
    return strip_history_suffix(feature).startswith("event_")


def is_quality_or_observation_feature(feature: str) -> bool:
    base = strip_history_suffix(feature)
    return base.startswith("obs_") or base.startswith("target_") or base.endswith("_complete")


def active_target_families(targets: Iterable[str]) -> set[str]:
    return {family for target in targets if (family := tim_family(target))}


def is_direct_target_leak(feature: str, targets: Iterable[str]) -> bool:
    base = strip_history_suffix(feature)
    history = is_history_feature(feature)
    target_set = set(targets)
    if base in target_set:
        return not history

    families = active_target_families(target_set)
    family = tim_family(base)
    if family in families and not history:
        return True

    if target_set.intersection(FLOW_TARGETS) and not history:
        if base in FLOW_LEAKAGE_COLUMNS:
            return True
        if base.startswith(FLOW_COMPONENT_PREFIXES):
            return True

    return False


def is_active_target_history(feature: str, targets: Iterable[str]) -> bool:
    if not is_history_feature(feature):
        return False
    base = strip_history_suffix(feature)
    target_set = set(targets)
    if base in target_set:
        return True
    family = tim_family(base)
    return family is not None and family in active_target_families(target_set)


def feature_group(feature: str, targets: Iterable[str]) -> str:
    if is_active_target_history(feature, targets):
        return "target_history"
    if is_calendar_feature(feature):
        return "calendar"
    if is_meteo_feature(feature):
        return "meteo"
    if is_event_feature(feature):
        return "events"
    if is_tim_feature(feature):
        return "tim"
    if is_pedoni_feature(feature):
        return "pedoni"
    return "other"


def allowed_groups_for_scope(scope: str) -> set[str]:
    if scope == "calendar_meteo_tim":
        return {"calendar", "meteo", "events", "tim", "target_history"}
    if scope == "calendar_meteo_pedoni":
        return {"calendar", "meteo", "events", "pedoni", "target_history"}
    if scope == "calendar_meteo_tim_pedoni":
        return {"calendar", "meteo", "events", "tim", "pedoni", "target_history"}
    if scope == "calendar_meteo":
        return {"calendar", "meteo", "events", "target_history"}
    raise ValueError(f"Feature scope non riconosciuto: {scope}")


def candidate_feature_allowed(
    feature: str,
    targets: Iterable[str],
    target_set: TargetSet,
    mode: str,
    feature_scope: str,
    allow_target_history: bool,
) -> bool:
    scope = resolved_feature_scope(target_set, feature_scope)
    if is_quality_or_observation_feature(feature):
        return False
    if is_direct_target_leak(feature, targets):
        return False

    group = feature_group(feature, targets)
    if group == "target_history" and not allow_target_history:
        return False
    if group not in allowed_groups_for_scope(scope):
        return False

    if mode == "forecast":
        # In forecast escludiamo TIM e varchi dello stesso orario del target.
        # Ma permettiamo i lag (dati storici), che sono validi anche in forecast.
        if group in {"tim", "pedoni"} and not is_history_feature(feature):
            return False

    return True


def historical_source_columns(df_columns: Iterable[str], targets: Iterable[str]) -> list[str]:
    source_cols = []
    target_set = set(targets)
    for col in df_columns:
        if col == "timestamp" or col in CALENDAR_FEATURE_COLUMNS:
            continue
        if is_history_feature(col) or is_quality_or_observation_feature(col):
            continue
        if col in target_set:
            continue
        if is_tim_feature(col) or is_meteo_feature(col) or is_pedoni_feature(col) or is_event_feature(col):
            source_cols.append(col)
    return sorted(set(source_cols))
