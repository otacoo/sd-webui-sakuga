from enum import Enum


class PresetArch(Enum):
    none = 0
    sd15 = 1
    sdxl = 2
    flux = 3
    sd3 = 4
    chroma = 5
    lumina = 6
    zit = 7
    wan = 8
    anima = 9

    @staticmethod
    def choices() -> list[str]:
        return [arch.name for arch in PresetArch]

    @staticmethod
    def default() -> "PresetArch":
        return PresetArch.none


SAMPLERS = {
    PresetArch.flux: "Euler",
    PresetArch.sd3: "Euler",
    PresetArch.chroma: "Euler",
    PresetArch.lumina: "Res Multistep",
    PresetArch.zit: "Euler",
    PresetArch.wan: "Euler",
    PresetArch.anima: "ER SDE",
}

SCHEDULERS = {
    PresetArch.flux: "Simple",
    PresetArch.sd3: "Simple",
    PresetArch.chroma: "Simple",
    PresetArch.lumina: "Simple",
    PresetArch.zit: "Beta",
    PresetArch.wan: "Simple",
    PresetArch.anima: "Beta",
}

STEPS = {
    PresetArch.flux: 28,
    PresetArch.sd3: 28,
    PresetArch.chroma: 4,
    PresetArch.lumina: 32,
    PresetArch.zit: 9,
    PresetArch.wan: 4,
    PresetArch.anima: 32,
}

CFG = {
    PresetArch.flux: 1.0,
    PresetArch.sd3: 5.0,
    PresetArch.chroma: 1.0,
    PresetArch.lumina: 4.0,
    PresetArch.zit: 1.0,
    PresetArch.wan: 5.0,
    PresetArch.anima: 4.0,
}
