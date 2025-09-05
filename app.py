import streamlit as st
import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary, PULP_CBC_CMD

# ===================================
# Funzioni ottimizzazione
# ===================================
def ottimizza_titolari(df, budget_totale, budget_ruoli, ruoli_richiesti, exclude_solutions, lambda_penalty, vincoli=[], soglia_titolare=85, exclude_players=[]):
    df_titolari = df[df["Titolare"] >= soglia_titolare].reset_index(drop=True)
    
    if exclude_players:
        df_titolari = df_titolari[~df_titolari["Nome"].isin(exclude_players)].reset_index(drop=True)

    prob = LpProblem("Titolari", LpMaximize)
    player_vars = {i: LpVariable(f"T_{i}", cat=LpBinary) for i in df_titolari.index}

    prob += lpSum(
        lambda_penalty * df_titolari.loc[i, "Media"] * player_vars[i] +
        df_titolari.loc[i, "Prezzo"] * player_vars[i] / budget_totale
        for i in df_titolari.index
    )

    # Vincoli numero giocatori per ruolo
    for ruolo, count in ruoli_richiesti.items():
        prob += lpSum(player_vars[i] for i in df_titolari.index if df_titolari.loc[i, "Ruolo"] == ruolo) == count

    # Vincoli budget per ruolo
    for ruolo, budget in budget_ruoli.items():
        prob += lpSum(df_titolari.loc[i, "Prezzo"] * player_vars[i] for i in df_titolari.index if df_titolari.loc[i, "Ruolo"] == ruolo) <= budget

    # Vincolo budget totale rigido
    prob += lpSum(df_titolari.loc[i, "Prezzo"] * player_vars[i] for i in df_titolari.index) <= budget_totale

    # Escludi soluzioni precedenti
    for prev in exclude_solutions:
        prob += lpSum(player_vars[i] for i in prev) <= len(prev) - 1

    # Vincoli giocatori obbligatori
    for nome in vincoli:
        idx = df_titolari.index[df_titolari["Nome"] == nome].tolist()
        if idx:
            prob += player_vars[idx[0]] == 1

    # Max 1 giocatore per squadra per ruolo
    for ruolo in df_titolari["Ruolo"].unique():
        squadre = df_titolari[df_titolari["Ruolo"]==ruolo]["Squadra"].unique()
        for squadra in squadre:
            idx_squadra = df_titolari.index[(df_titolari["Ruolo"]==ruolo) & (df_titolari["Squadra"]==squadra)].tolist()
            prob += lpSum(player_vars[i] for i in idx_squadra) <= 1

    prob.solve(PULP_CBC_CMD(msg=0))
    if prob.status != 1:
        return None
    selezionati = [i for i in df_titolari.index if player_vars[i].value() == 1]
    df_result = df_titolari.loc[selezionati].copy()
    df_result["Vincolato"] = df_result["Nome"].apply(lambda x: "✅" if x in vincoli else "")
    return df_result

def ottimizza_riserve(df, budget_totale, budget_ruoli, ruoli_richiesti, exclude_solutions, titolari, lambda_penalty, vincoli=[], soglia_titolare=85, exclude_players=[]):
    df_riserve = df[~df["Nome"].isin(titolari["Nome"])].reset_index(drop=True)
    
    if exclude_players:
        df_riserve = df_riserve[~df_riserve["Nome"].isin(exclude_players)].reset_index(drop=True)

    prob = LpProblem("Riserve", LpMaximize)
    player_vars = {i: LpVariable(f"R_{i}", cat=LpBinary) for i in df_riserve.index}

    # Obiettivo: budget first, titolarità second
    prob += lpSum(
        lambda_penalty * df_riserve.loc[i, "Titolare"] * player_vars[i] +
        df_riserve.loc[i, "Prezzo"] * player_vars[i] / budget_totale
        for i in df_riserve.index
    )

    # Vincoli numero giocatori per ruolo
    for ruolo, count in ruoli_richiesti.items():
        prob += lpSum(player_vars[i] for i in df_riserve.index if df_riserve.loc[i, "Ruolo"] == ruolo) == count

    # Vincoli budget per ruolo
    for ruolo, budget in budget_ruoli.items():
        prob += lpSum(df_riserve.loc[i, "Prezzo"] * player_vars[i] for i in df_riserve.index if df_riserve.loc[i, "Ruolo"] == ruolo) <= budget

    # Vincolo budget totale rigido
    prob += lpSum(df_riserve.loc[i, "Prezzo"] * player_vars[i] for i in df_riserve.index) <= budget_totale

    # Escludi soluzioni precedenti
    for prev in exclude_solutions:
        prob += lpSum(player_vars[i] for i in prev) <= len(prev) - 1

    # Vincoli giocatori obbligatori
    for nome in vincoli:
        idx = df_riserve.index[df_riserve["Nome"] == nome].tolist()
        if idx:
            prob += player_vars[idx[0]] == 1

    prob.solve(PULP_CBC_CMD(msg=0))
    if prob.status != 1:
        return None
    selezionati = [i for i in df_riserve.index if player_vars[i].value() == 1]
    df_result = df_riserve.loc[selezionati].copy()
    df_result["Vincolato"] = df_result["Nome"].apply(lambda x: "✅" if x in vincoli else "")
    return df_result

# ===================================
# Funzione riadattamento budget
# ===================================
def riadatta_budget(budget_totale, perc, df, vincoli):
    budget_ruoli = {r: (perc[r]/100)*budget_totale for r in perc}
    ruoli_sforati = []
    for ruolo in budget_ruoli:
        vincolati_ruolo = df[(df["Nome"].isin(vincoli)) & (df["Ruolo"]==ruolo)]
        spesa_vincolati = vincolati_ruolo["Prezzo"].sum()
        if spesa_vincolati > budget_ruoli[ruolo]:
            budget_ruoli[ruolo] = spesa_vincolati
            ruoli_sforati.append(ruolo)
    totale_usato = sum(budget_ruoli.values())
    rimanente = budget_totale - totale_usato
    altri_ruoli = [r for r in budget_ruoli if r not in ruoli_sforati]
    if rimanente > 0 and altri_ruoli:
        aggiunta = rimanente / len(altri_ruoli)
        for r in altri_ruoli:
            budget_ruoli[r] += aggiunta
    return budget_ruoli

# ===================================
# Streamlit App
# ===================================
st.title("⚽ Fantacalcio Optimizer - Completo")

uploaded_file = st.file_uploader("Carica il file Excel con i giocatori", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("📊 Anteprima dati:", df.head())

    st.subheader("Parametri generali")
    budget_totale_titolari = st.number_input("Budget totale Titolari", value=450)
    budget_totale_riserve = st.number_input("Budget totale Riserve", value=50)
    num_alternative = st.slider("Numero di alternative", 1, 10, 5)
    lambda_penalty = st.number_input("Peso incentivo a spendere / titolarità (λ)", value=0.001, format="%.4f")
    soglia_titolare = st.number_input("Soglia titolarità minima", value=85, min_value=0, max_value=100, step=1)

    st.subheader("Percentuali budget per ruolo")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Titolari**")
        perc_titolari = {
            "P": st.number_input("P (Portieri)", value=8.5, step=0.1, key="titolari_P"),
            "D": st.number_input("D (Difensori)", value=16.0, step=0.1, key="titolari_D"),
            "C": st.number_input("C (Centrocampisti)", value=23.5, step=0.1, key="titolari_C"),
            "A": st.number_input("A (Attaccanti)", value=52.0, step=0.1, key="titolari_A"),
        }
    with col2:
        st.markdown("**Riserve**")
        perc_riserve = {
            "P": st.number_input("P (Portieri)", value=8.5, step=0.1, key="riserve_P"),
            "D": st.number_input("D (Difensori)", value=16.0, step=0.1, key="riserve_D"),
            "C": st.number_input("C (Centrocampisti)", value=23.5, step=0.1, key="riserve_C"),
            "A": st.number_input("A (Attaccanti)", value=52.0, step=0.1, key="riserve_A"),
        }

    st.subheader("Giocatori vincolanti")
    giocatori_obbligatori = st.multiselect("Seleziona giocatori da includere", df["Nome"].tolist())

    st.subheader("Giocatori da escludere")
    giocatori_da_escludere = st.multiselect("Seleziona giocatori da escludere", df["Nome"].tolist())

    # Riadatta budget in caso di vincolati
    budget_titolari = riadatta_budget(budget_totale_titolari, perc_titolari, df, giocatori_obbligatori)
    budget_riserve  = riadatta_budget(budget_totale_riserve, perc_riserve, df, giocatori_obbligatori)

    ruoli_titolari = {"P": 1, "D": 4, "C": 3, "A": 3}
    ruoli_riserve  = {"P": 2, "D": 4, "C": 5, "A": 3}

    if st.button("Genera squadre"):
        titolari_solutions = []
        riserve_solutions = []

        for k in range(num_alternative):
            titolari = ottimizza_titolari(
                df, budget_totale_titolari, budget_titolari, ruoli_titolari,
                titolari_solutions, lambda_penalty, vincoli=giocatori_obbligatori,
                soglia_titolare=soglia_titolare, exclude_players=giocatori_da_escludere
            )
            if titolari is None:
                st.warning(f"Nessuna soluzione titolari alla iterazione {k+1}")
                break
            titolari_idx = list(titolari.index)
            titolari_solutions.append(titolari_idx)

            riserve = ottimizza_riserve(
                df, budget_totale_riserve, budget_riserve, ruoli_riserve,
                riserve_solutions, titolari, lambda_penalty, vincoli=giocatori_obbligatori,
                soglia_titolare=soglia_titolare, exclude_players=giocatori_da_escludere
            )
            if riserve is None:
                st.warning(f"Nessuna soluzione riserve alla iterazione {k+1}")
                break
            riserve_idx = list(riserve.index)
            riserve_solutions.append(riserve_idx)

            st.markdown(f"## ⚡ Alternativa {k+1}")
            st.markdown("### 🏆 Titolari")
            st.dataframe(titolari)
            st.write("Prezzo totale:", titolari["Prezzo"].sum(), " | Media totale:", titolari["Media"].sum())
            st.markdown("### 🔄 Riserve")
            st.dataframe(riserve)
            st.write("Prezzo totale:", riserve["Prezzo"].sum(), " | Indice Titolarità totale:", riserve["Titolare"].sum())
