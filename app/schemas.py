"""Modelli Pydantic v2: input, output strutturato e ricette."""
from typing import Literal

from pydantic import BaseModel, Field

Unit = Literal["g", "ml", "pz"]
Category = Literal["Ortofrutta", "Pescheria", "Dispensa", "Latticini"]


class Ingredient(BaseModel):
    name: str
    quantity: float = Field(gt=0)
    unit: Unit
    category: Category
    is_perishable: bool


class Recipe(BaseModel):
    id: str
    title: str
    prep_time_min: int
    season_months: list[int]
    diet_tags: list[str]
    calories_per_portion: int
    ingredients: list[Ingredient]
    instructions: str
    # Porzioni a cui si riferiscono le quantità degli ingredienti nel dataset.
    base_portions: int = Field(default=2, ge=1)


class WeeklyMenuRequest(BaseModel):
    days: int = Field(default=7, ge=1, le=14)
    portions: int = Field(default=2, ge=1, le=12)
    current_month: int = Field(ge=1, le=12)
    fridge_ingredients: list[str] = Field(default_factory=list)
    dietary_restrictions: list[str] = Field(default_factory=list)


class MealEntry(BaseModel):
    recipe_id: str
    recipe_title: str


class DayMenu(BaseModel):
    day: int
    lunch: MealEntry
    dinner: MealEntry


class ShoppingListItem(BaseModel):
    name: str
    total_quantity: float
    unit: str
    category: str


class WeeklyMenuResponse(BaseModel):
    menu: list[DayMenu]
    # Il LLM può lasciarla vuota: viene sempre ricalcolata in Python.
    shopping_list: list[ShoppingListItem] = Field(default_factory=list)
    anti_waste_notes: str
    # Ricette usate nel menu, con quantità già scalate sulle porzioni richieste.
    recipes: list[Recipe] = Field(default_factory=list)

# ---- Fridge rescue -------------------------------------------------------
class RescueTip(BaseModel):
    recipe_id: str
    tips: str


class RescueTipsLLM(BaseModel):
    suggestions: list[RescueTip]


class RescueResult(BaseModel):
    recipe: Recipe
    suggestions: str


class FridgeRescueResponse(BaseModel):
    results: list[RescueResult]

# ---- Piano compatto prodotto dal LLM (solo id: molto più veloce) ----------
class MenuPlanDay(BaseModel):
    lunch_id: str
    dinner_id: str


class MenuPlanLLM(BaseModel):
    days: list[MenuPlanDay]
    anti_waste_notes: str