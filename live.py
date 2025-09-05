import streamlit as st
import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary, PULP_CBC_CMD

# =======================
# Funzione ottimizzazione giocatori
# =======================
def ottimizza_giocatori(df, budget_totale, budget_ruoli, ruoli_richiesti, exclude_solutions, lambda_penalty, vincoli=[], soglia_titolare=85, exclude_players=[]):
    df_filtrato = df[df["Titolare"] >= soglia_titolare].reset_index(drop=True)
    if exclude_players:
        df_filtrato = df_filtrato[~df_filtrato["Nome"].isin(exclude_players)].reset_index(drop=True)

    prob = LpProblem("Asta", LpMaximize)
    player_vars = {i: LpVariable(f"G_{i}", cat=LpBinary) for i in df_filtrato.index}

    # Obiettivo: massimizzare media ponderata e minimizzare prezzo
    prob += lpSum(
        lambda_penalty * df_filtrato.loc[i, "Media"] * player_vars[i] -
        df_filtrato.loc[i, "Prezzo"] * player_vars[i] / budget_totale
        for i in df_filtrato.index
    )

    # Vincoli numero giocatori per ruolo
    for ruolo, count in ruoli_richiesti.items():
        prob += lpSum(player_vars[i] for i in df_filtrato.index if df_filtrato.loc[i, "Ruolo"] == ruolo) == count

    # Vincoli budget per ruolo
    for ruolo, budget in budget_ruoli.items():
        prob += lpSum(df_filtrato.loc[i, "Prezzo"] * player_vars[i] for i in df_filtrato.index if df_filtrato.loc[i, "Ruolo"] == ruolo) <= budget

    # Vincolo budget totale
    prob += lpSum(df_filtrato.loc[i, "Prezzo"] * player_vars[i] for i in df_filtrato.index) <= budget_totale

    # Escludi soluzioni precedenti
    for prev in exclude_solutions:
        prob += lpSum(player_vars[i] for i in prev) <= len(prev) - 1

    # Vincoli giocatori obbligatori
    for nome in vincoli:
        idx = df_filtrato.index[df_filtrato["Nome"] == nome].tolist()
        if idx:
            prob += player_vars[idx[0]] == 1

    prob.solve(PULP_CBC_CMD(msg=0))
    if prob.status != 1:
        return None

    selezionati = [i for i in df_filtrato.index if player_vars[i].value() == 1]
    df_result = df_filtrato.loc[selezionati].copy()
    df_result["Vincolato"] = df_result["Nome"].apply(lambda x: "✅" if x in vincoli else "")
    return df_result

# =======================
# Streamlit - interfaccia d'asta live
# =======================
st.title("⚽ Fantacalcio Asta Optimizer Live")

uploaded_file = st.file_uploader("Carica file Excel con giocatori", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("📊 Anteprima dati", df.head())

    st.subheader("Parametri asta")
    budget_totale_iniziale = st.number_input("Budget totale", value=500)
    lambda_penalty = st.number_input("Peso ottimizzazione (λ)", value=0.001, format="%.4f")
    soglia_titolare = st.number_input("Soglia titolarità minima", value=85, min_value=0, max_value=100)

    st.subheader("Composizione ruoli")
    ruoli_richiesti = {
        "P": st.number_input("P (Portieri)", value=3),
        "D": st.number_input("D (Difensori)", value=8),
        "C": st.number_input("C (Centrocampisti)", value=8),
        "A": st.number_input("A (Attaccanti)", value=6),
    }

    # Stato live: budget residuo e giocatori acquistati
    if 'budget_residuo' not in st.session_state:
        st.session_state.budget_residuo = budget_totale_iniziale
    if 'acquistati' not in st.session_state:
        st.session_state.acquistati = []

    st.subheader("Giocatori già acquistati")
    nuovi_acquisti = st.multiselect("Seleziona giocatori acquistati in asta", df["Nome"].tolist(), default=st.session_state.acquistati)
    st.session_state.acquistati = nuovi_acquisti
    spesa_totale = df[df["Nome"].isin(st.session_state.acquistati)]["Prezzo"].sum()
    st.session_state.budget_residuo = budget_totale_iniziale - spesa_totale
    st.write(f"💰 Budget residuo: {st.session_state.budget_residuo}")

    st.subheader("Giocatori vincolati")
    vincolati = st.multiselect("Seleziona giocatori da acquistare a tutti i costi", df["Nome"].tolist())

    st.subheader("Suggerimenti prossimi acquisti")
    num_alternative = st.slider("Numero di alternative", 1, 5, 3)

    if st.button("Genera consigli"): 
        budget_ruoli = {r: (1/len(ruoli_richiesti))*st.session_state.budget_residuo for r in ruoli_richiesti}
        soluzioni = []
        for k in range(num_alternative):
            suggerimento = ottimizza_giocatori(
                df, st.session_state.budget_residuo, budget_ruoli, ruoli_richiesti,
                soluzioni, lambda_penalty, vincoli=vincolati, soglia_titolare=soglia_titolare,
                exclude_players=st.session_state.acquistati
            )
            if suggerimento is None:
                st.warning(f"Nessuna soluzione trovata per alternativa {k+1}")
                break
            soluzioni.append(list(suggerimento.index))
            st.markdown(f"### Alternativa {k+1}")
            st.dataframe(suggerimento)
            st.write("Prezzo totale:", suggerimento["Prezzo"].sum(), "| Media totale:", suggerimento["Media"].sum())
