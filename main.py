# main.py
from typing import Optional, List, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Cattle Methane Reduction API", version="1.0.0")

# CORS (adjust origins for your company domains if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Constants (match your final app.py)
# -----------------------------
EMISSION_FACTORS_KG_PER_HEAD_YR = {
    "cow": 72.0,      # kg CH4/head·year
    "buffalo": 90.0,  # kg CH4/head·year
}

# Diet reduction (conservative)
DIET_REDUCTION = {
    "conventional": 0.00,
    "improved": 0.08,
    "high-quality": 0.12,
}

# Additives (conservative + India)
ADDITIVE_REDUCTION = {
    "none": 0.00,
    "harit dhara (icar)": 0.18,
    "seaweed": 0.25,
    "3-NOP": 0.30,
    "oils": 0.08,
}

# Fixed GWP (AR6)
GWP_VALUE = 27.2  # IPCC AR6 (Latest Update)

# Equivalences
TREE_T_CO2E_PER_YEAR = 0.021
CAR_T_CO2E_PER_YEAR = 4.6

# Tier-2 intake assumptions (conservative)
DMI_PCT_BY_DIET = {
    "conventional": 0.019,  # 1.9% BW/day
    "improved": 0.022,      # 2.2% BW/day
    "high-quality": 0.025,  # 2.5% BW/day
}
# Ym by diet
YM_BY_DIET = {
    "conventional": 7.0,
    "improved": 6.5,
    "high-quality": 6.0,
}
GE_DENSITY_MJ_PER_KG_DM = 18.45
CH4_ENERGY_MJ_PER_KG = 55.65

# -----------------------------
# Helpers (same math as app.py)
# -----------------------------
def combined_reduction_fraction(f_diet: float, f_add: float) -> float:
    return 1 - (1 - f_diet) * (1 - f_add)

def baseline_tCH4(ef_kg_per_head_yr: float, n_animals: int) -> float:
    return (ef_kg_per_head_yr * n_animals) / 1000.0

def calc_dynamic_ef_kg_per_head_yr(weight_kg: float, diet: str) -> float:
    """Tier-2 style EF from weight + diet."""
    if weight_kg is None or weight_kg <= 0:
        return 0.0
    dmi_pct = DMI_PCT_BY_DIET.get(diet, 0.019)
    ym = YM_BY_DIET.get(diet, 7.0)
    dmi_kg_day = weight_kg * dmi_pct
    ge_mj_day = dmi_kg_day * GE_DENSITY_MJ_PER_KG_DM
    ch4_energy_mj_day = ge_mj_day * (ym / 100.0)
    kg_ch4_day = ch4_energy_mj_day / CH4_ENERGY_MJ_PER_KG
    ef_kg_yr = kg_ch4_day * 365.0
    return ef_kg_yr

def compute_results(
    n: int,
    animal_type: str,
    diet: str,
    additive: str,
    ef_override_kg_per_head_yr: Optional[float] = None,
):
    # choose EF: dynamic if provided, else defaults
    if ef_override_kg_per_head_yr and ef_override_kg_per_head_yr > 0:
        ef = ef_override_kg_per_head_yr
    else:
        ef = EMISSION_FACTORS_KG_PER_HEAD_YR[animal_type]

    f_diet = DIET_REDUCTION[diet]
    f_add = ADDITIVE_REDUCTION[additive]

    base_tCH4 = baseline_tCH4(ef, n)
    f_total = combined_reduction_fraction(f_diet, f_add)
    reduced_tCH4 = base_tCH4 * f_total
    avoided_tCO2e = reduced_tCH4 * GWP_VALUE

    cars = avoided_tCO2e / CAR_T_CO2E_PER_YEAR
    trees = avoided_tCO2e / TREE_T_CO2E_PER_YEAR
    base_tCO2e = base_tCH4 * GWP_VALUE

    return {
        "baseline_tCH4": base_tCH4,
        "baseline_tCO2e": base_tCO2e,
        "methane_reduced_tCH4": reduced_tCH4,
        "avoided_tCO2e": avoided_tCO2e,
        "cars_removed": cars,
        "trees_equivalent": trees,
        "emission_factor_used": ef,
        "gwp_used": GWP_VALUE,
        "diet_reduction_fraction": f_diet,
        "additive_reduction_fraction": f_add,
        "total_reduction_fraction": f_total,
    }

def compute_what_if_rows(
    n: int,
    animal_type: str,
    diet: str,
    ef_override_kg_per_head_yr: Optional[float] = None,
):
    if ef_override_kg_per_head_yr and ef_override_kg_per_head_yr > 0:
        ef = ef_override_kg_per_head_yr
    else:
        ef = EMISSION_FACTORS_KG_PER_HEAD_YR[animal_type]

    base_tCH4 = baseline_tCH4(ef, n)
    f_diet = DIET_REDUCTION[diet]

    rows = []
    for add_name, f_add in ADDITIVE_REDUCTION.items():
        if add_name == "none":
            continue
        f_total = combined_reduction_fraction(f_diet, f_add)
        tCH4_red = base_tCH4 * f_total
        tCO2e = tCH4_red * GWP_VALUE
        rows.append(
            {
                "additive": add_name,
                "reduction_percent": round(f_total * 100, 2),
                "methane_reduced_tCH4": tCH4_red,
                "avoided_tCO2e": tCO2e,
                "cars_removed": tCO2e / CAR_T_CO2E_PER_YEAR,
                "trees_equivalent": tCO2e / TREE_T_CO2E_PER_YEAR,
            }
        )
    rows.sort(key=lambda r: r["avoided_tCO2e"], reverse=True)
    return base_tCH4, base_tCH4 * GWP_VALUE, rows

# -----------------------------
# Schemas
# -----------------------------
AnimalType = Literal["cow", "buffalo"]
DietType = Literal["conventional", "improved", "high-quality"]
AdditiveType = Literal["none", "harit dhara (icar)", "seaweed", "3-NOP", "oils"]

class CalculateRequest(BaseModel):
    num_animals: int = Field(..., gt=0)
    animal_type: AnimalType
    diet: DietType
    additive: AdditiveType
    weight: Optional[float] = Field(None, gt=0)

    @validator("animal_type")
    def validate_animal(cls, v):
        if v not in EMISSION_FACTORS_KG_PER_HEAD_YR:
            raise ValueError("animal_type must be one of: cow, buffalo")
        return v

class WhatIfRequest(BaseModel):
    num_animals: int = Field(..., gt=0)
    animal_type: AnimalType
    diet: DietType
    weight: Optional[float] = Field(None, gt=0)

    @validator("animal_type")
    def validate_animal(cls, v):
        if v not in EMISSION_FACTORS_KG_PER_HEAD_YR:
            raise ValueError("animal_type must be one of: cow, buffalo")
        return v

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Cattle Methane Reduction API is running.",
        "endpoints": ["/calculate (POST)", "/what-if (POST)"],
        "gwp_used": GWP_VALUE,
    }

@app.post("/calculate")
def calculate(req: CalculateRequest):
    # Dynamic EF if weight provided
    ef_dynamic = calc_dynamic_ef_kg_per_head_yr(req.weight, req.diet) if req.weight else None

    result = compute_results(
        n=req.num_animals,
        animal_type=req.animal_type,
        diet=req.diet,
        additive=req.additive,
        ef_override_kg_per_head_yr=ef_dynamic,
    )

    # Round final numbers for cleaner JSON
    def r(x, nd=4):
        return None if x is None else round(float(x), nd)

    return {
        "inputs": {
            "num_animals": req.num_animals,
            "animal_type": req.animal_type,
            "diet": req.diet,
            "additive": req.additive,
            "weight": req.weight,
        },
        "results": {
            "baseline_tCH4": r(result["baseline_tCH4"]),
            "baseline_tCO2e": r(result["baseline_tCO2e"]),
            "methane_reduced_tCH4": r(result["methane_reduced_tCH4"]),
            "avoided_tCO2e": r(result["avoided_tCO2e"]),
            "cars_removed": r(result["cars_removed"]),
            "trees_equivalent": r(result["trees_equivalent"]),
            "emission_factor_used": r(result["emission_factor_used"]),
            "diet_reduction_fraction": r(result["diet_reduction_fraction"]),
            "additive_reduction_fraction": r(result["additive_reduction_fraction"]),
            "total_reduction_fraction": r(result["total_reduction_fraction"]),
            "gwp_used": GWP_VALUE,
            "ef_method": "weight-based" if ef_dynamic else "default",
        },
    }

@app.post("/what-if")
def what_if(req: WhatIfRequest):
    ef_dynamic = calc_dynamic_ef_kg_per_head_yr(req.weight, req.diet) if req.weight else None
    base_tCH4, base_tCO2e, rows = compute_what_if_rows(
        n=req.num_animals,
        animal_type=req.animal_type,
        diet=req.diet,
        ef_override_kg_per_head_yr=ef_dynamic,
    )

    def r(x, nd=4):
        return None if x is None else round(float(x), nd)

    # Round within rows
    cleaned_rows = [
        {
            "additive": row["additive"],
            "reduction_percent": r(row["reduction_percent"], 2),
            "methane_reduced_tCH4": r(row["methane_reduced_tCH4"]),
            "avoided_tCO2e": r(row["avoided_tCO2e"]),
            "cars_removed": r(row["cars_removed"]),
            "trees_equivalent": r(row["trees_equivalent"]),
        }
        for row in rows
    ]

    return {
        "inputs": {
            "num_animals": req.num_animals,
            "animal_type": req.animal_type,
            "diet": req.diet,
            "weight": req.weight,
        },
        "baseline": {
            "baseline_tCH4": r(base_tCH4),
            "baseline_tCO2e": r(base_tCO2e),
            "emission_factor_used": r(
                calc_dynamic_ef_kg_per_head_yr(req.weight, req.diet)
                if req.weight else EMISSION_FACTORS_KG_PER_HEAD_YR[req.animal_type]
            ),
            "gwp_used": GWP_VALUE,
            "ef_method": "weight-based" if ef_dynamic else "default",
        },
        "recommendations": cleaned_rows,
    }
