from enum import StrEnum


class UnitEnum(StrEnum):
    # subclass contract: members are defined as (token, to_base, aliases) and have a default classmethod
    def __new__(cls, token: str, to_base: float, aliases=()):
        obj = str.__new__(cls, token)
        obj._value_ = token
        obj.to_base = float(to_base)
        obj.aliases = tuple(a.lower() for a in aliases)
        return obj

    @classmethod
    def labels(cls) -> list:
        return [m.value for m in cls]

    @classmethod
    def from_token(cls, token):
        t = str(token).strip().lower()
        for u in cls:
            if t == u.value or t in u.aliases:
                return u
        raise ValueError(f"Unknown unit {token!r}. Valid units are {cls.labels()}")

    @classmethod
    def from_any(cls, arg):
        if arg is None:
            return cls.default
        if isinstance(arg, cls):
            return arg
        return cls.from_token(arg)


class LengthUnits(UnitEnum):
    METERS = ("meters", 1.0, ("m", "meter"))
    FEET = ("feet", 0.3048, ("ft", "foot"))
    INCHES = ("inches", 0.0254, ("in", "inch"))
    CENTIMETERS = ("centimeters", 0.01, ("cm", "centimeter"))
    YARDS = ("yards", 0.9144, ("yard",))

    @classmethod
    def default(cls):
        return cls.METERS


class TimeUnits(UnitEnum):
    SECONDS = ("seconds", 1.0, ("s", "secs", "sec"))
    MINUTES = ("minutes", 60.0, ("min", "mins"))
    HOURS = ("hours", 3600.0, ("hour", "hr"))
    DAYS = ("days", 86400.0, ("d", "days"))

    @classmethod
    def default(cls):
        return cls.SECONDS


def convert(
    enum,
    src,
    dst,
    *args,
    sigfigs=None,
    default_src=None,
    default_dst=None,
):
    s = enum.from_any(src)
    d = enum.from_any(dst)
    if s == d:
        return args[0] if len(args) == 1 else args

    factor = s.to_base / d.to_base
    out = tuple(None if a is None else a * factor for a in args)
    if sigfigs is not None:
        out = tuple(None if a is None else round(a, sigfigs) for a in out)
    return out[0] if len(out) == 1 else out


def convert_length(src, dst, *args, sigfigs: int | None = 12):
    return convert(LengthUnits, src, dst, *args, sigfigs=sigfigs)


def convert_time(src, dst, *args, sigfigs: int | None = None):
    return convert(TimeUnits, src, dst, *args, sigfigs=sigfigs)


def convert_units(src, dst, *args, sigfigs: int | None = 12, unit_type="length"):
    if unit_type.lower() == "length":
        return convert_length(src, dst, *args, sigfigs=sigfigs)
    elif unit_type.lower() == "time":
        return convert_time(src, dst, *args, sigfigs=sigfigs)
    raise ValueError(f"{unit_type} is not a valid unit type")
