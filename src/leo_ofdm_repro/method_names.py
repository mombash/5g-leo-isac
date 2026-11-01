"""Public method names and compatibility identifiers for saved research records."""

# Serialized records use these identifiers. Public commands accept both columns.
METHOD_NAMES = {
    'N5': 'neural',
    'S5': 'statistical',
    'C9': 'adaptive-classical',
    'E27': 'exhaustive',
    'DS': 'dual-scale',
    'TS1': 'two-step-fixed',
    'TS2': 'two-step-early',
    'KT+LVD': 'hybrid-integration',
    'N5_always27': 'neural-full-final',
    'S5_always27': 'statistical-full-final',
    'C9_always27': 'adaptive-classical-full-final',
}


def internal_name(name):
    if name in METHOD_NAMES:
        return name
    for identifier, public in METHOD_NAMES.items():
        if name == public:
            return identifier
    raise ValueError('Unknown method: ' + str(name))


def public_name(name):
    return METHOD_NAMES[internal_name(name)]
