const API = "/api/v1";
const MESI = [
  "Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno",
  "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre",
];
const CORSIE = ["Ortofrutta", "Pescheria", "Latticini", "Dispensa"];

const $ = (id) => document.getElementById(id);

// Crea un elemento con testo sicuro (textContent, mai HTML dalla risposta del modello)
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function numero(n) {
  return Number(n).toLocaleString("it-IT", { maximumFractionDigits: 2 });
}

function dividiLista(testo) {
  return testo.split(",").map((s) => s.trim()).filter(Boolean);
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let dettaglio = "Errore " + res.status;
    try {
      const data = await res.json();
      if (data.detail) {
        dettaglio = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
      }
    } catch (e) { /* risposta non JSON */ }
    throw new Error(dettaglio);
  }
  return res.json();
}

// ---------- Schede ----------
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $(btn.dataset.target).classList.add("active");
  });
});

// ---------- Mese predefinito ----------
MESI.forEach((nome, i) => {
  const opt = el("option", "", nome);
  opt.value = i + 1;
  $("month").appendChild(opt);
});
$("month").value = new Date().getMonth() + 1;

// ---------- Menu settimanale ----------
$("menu-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = $("menu-btn");
  const status = $("menu-status");
  const result = $("menu-result");

  const restrizioni = [...document.querySelectorAll('input[name="restriction"]:checked')]
    .map((c) => c.value);

  const body = {
    days: Number($("days").value),
    portions: Number($("portions").value),
    current_month: Number($("month").value),
    fridge_ingredients: dividiLista($("fridge").value),
    dietary_restrictions: restrizioni,
  };

  btn.disabled = true;
  status.className = "status";
  status.textContent = "Sto preparando il menu... può richiedere qualche minuto, non chiudere la pagina.";
  result.replaceChildren();

  try {
    const data = await postJSON(API + "/generate-menu", body);
    status.textContent = "";
    mostraMenu(data, result);
  } catch (err) {
    status.className = "status error";
    status.textContent = err.message;
  } finally {
    btn.disabled = false;
  }
});

function mostraMenu(data, contenitore) {
  const ricette = {};
  (data.recipes || []).forEach((r) => { ricette[r.id] = r; });

  // Note anti-spreco
  contenitore.appendChild(el("h2", "", "♻️ Come evitiamo gli sprechi"));
  contenitore.appendChild(el("div", "notes", data.anti_waste_notes));

  // Giorni (i pasti sono cliccabili)
  contenitore.appendChild(el("h2", "", "🍽️ Menu (clicca un pasto per vedere la ricetta)"));
  const griglia = el("div", "days");
  data.menu.forEach((giorno) => {
    const card = el("div", "day");
    card.appendChild(el("h3", "", "Giorno " + giorno.day));
    [["Pranzo", giorno.lunch], ["Cena", giorno.dinner]].forEach(([etichetta, pasto]) => {
      const riga = el("div", "meal");
      riga.appendChild(el("small", "", etichetta));
      const bottone = el("button", "meal-btn", pasto.recipe_title);
      bottone.type = "button";
      bottone.addEventListener("click", () => {
        if (ricette[pasto.recipe_id]) apriRicetta(ricette[pasto.recipe_id]);
      });
      riga.appendChild(bottone);
      card.appendChild(riga);
    });
    griglia.appendChild(card);
  });
  contenitore.appendChild(griglia);

  // Lista della spesa, divisa per corsia
  contenitore.appendChild(el("h2", "", "🛒 Lista della spesa"));
  const corsie = [...new Set([...CORSIE, ...data.shopping_list.map((i) => i.category)])];
  corsie.forEach((corsia) => {
    const voci = data.shopping_list.filter((i) => i.category === corsia);
    if (voci.length === 0) return;
    const box = el("div", "shopping-cat");
    box.appendChild(el("h3", "", corsia));
    voci.forEach((voce) => {
      const label = el("label");
      const check = document.createElement("input");
      check.type = "checkbox";
      label.appendChild(check);
      label.appendChild(el("span", "", " " + voce.name + " — " + numero(voce.total_quantity) + " " + voce.unit));
      box.appendChild(label);
    });
    contenitore.appendChild(box);
  });
}

function apriRicetta(ricetta) {
  const finestra = document.createElement("dialog");
  finestra.className = "ricetta-dialog";

  finestra.appendChild(el("h3", "", ricetta.title));
  finestra.appendChild(el("div", "meta",
    ricetta.prep_time_min + " min · " + ricetta.calories_per_portion +
    " kcal a porzione · ingredienti per " + ricetta.base_portions + " porzioni"));

  const lista = el("ul");
  ricetta.ingredients.forEach((ing) => {
    lista.appendChild(el("li", "", ing.name + " — " + numero(ing.quantity) + " " + ing.unit));
  });
  finestra.appendChild(lista);
  finestra.appendChild(el("p", "", ricetta.instructions));

  const chiudi = el("button", "chiudi-btn", "Chiudi");
  chiudi.type = "button";
  chiudi.addEventListener("click", () => finestra.close());
  finestra.appendChild(chiudi);

  finestra.addEventListener("close", () => finestra.remove());
  document.body.appendChild(finestra);
  finestra.showModal();
}


// ---------- Salva-frigo ----------
$("rescue-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = $("rescue-btn");
  const status = $("rescue-status");
  const result = $("rescue-result");

  const ingredienti = dividiLista($("rescue-input").value);
  if (ingredienti.length < 2 || ingredienti.length > 4) {
    status.className = "status error";
    status.textContent = "Inserisci da 2 a 4 ingredienti, separati da virgola.";
    return;
  }

  btn.disabled = true;
  status.className = "status";
  status.textContent = "Cerco le ricette migliori... può richiedere qualche minuto.";
  result.replaceChildren();

  try {
    const data = await postJSON(API + "/fridge-rescue", ingredienti);
    status.textContent = "";
    mostraRicette(data.results, result);
  } catch (err) {
    status.className = "status error";
    status.textContent = err.message;
  } finally {
    btn.disabled = false;
  }
});

function mostraRicette(risultati, contenitore) {
  contenitore.appendChild(el("h2", "", "Ricette consigliate"));
  risultati.forEach(({ recipe, suggestions }) => {
    const card = el("div", "recipe");
    card.appendChild(el("h3", "", recipe.title));
    card.appendChild(el("div", "meta",
      recipe.prep_time_min + " min · " + recipe.calories_per_portion + " kcal a porzione"));

    const lista = el("ul");
    recipe.ingredients.forEach((ing) => {
      lista.appendChild(el("li", "", ing.name + " — " + numero(ing.quantity) + " " + ing.unit));
    });
    card.appendChild(lista);
    card.appendChild(el("p", "", recipe.instructions));
    card.appendChild(el("div", "tips", "💡 " + suggestions));
    contenitore.appendChild(card);
  });
}