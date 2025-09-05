import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression

# -----------------------------
# Configurazione pagina
# -----------------------------
st.set_page_config(page_title="Asta Master", layout="wide")

# -----------------------------
# Funzioni utili
# -----------------------------
def load_data_from_excel(buffer):
    return pd.read_excel(buffer)

def calcola_valore(df):
    df["Valore"] = df["Media"] / df["Prezzo"]
    df["Score"] = df["Media"] * (df["Titolare"] / 100) / df["Prezzo"]
    return df

def suggerisci_squadra(df, budget, slot_ruoli):
    squadra = []
    budget_residuo = budget
    for ruolo, n_slot in slot_ruoli.items():
        disponibili = df[df["Ruolo"] == ruolo].sort_values(by="Score", ascending=False)
        for _, row in disponibili.iterrows():
            if n_slot <= 0:
                break
            if row["Prezzo"] <= budget_residuo:
                squadra.append(row)
                budget_residuo -= row["Prezzo"]
                n_slot -= 1
    return pd.DataFrame(squadra), budget_residuo

# -----------------------------
# Titolo
# -----------------------------
st.title("🚀 Asta Master - Consulente Live per l'Asta del Fantacalcio")

# -----------------------------
# Upload o dataset di esempio
# -----------------------------
uploaded_file = st.file_uploader("📂 Carica il tuo file Excel dei giocatori", type=["xlsx"])

use_example = st.button("Usa dataset d'esempio (se incluso)")

if uploaded_file:
    df = load_data_from_excel(uploaded_file)
    st.success("✅ File Excel caricato con successo")
elif use_example:
    st.info("ℹ️ Dataset di esempio caricato.")
    df = pd.DataFrame({
        "Nome": ["ABOUKHLAL", "ACERBI", "ADAMS C."],
        "Squadra": ["TOR", "INT", "TOR"],
        "Ruolo": ["A", "D", "A"],
        "Prezzo": [1.45, 5.95, 8.65],
        "Media": [6.62, 6.16, 6.61],
        "Titolare": [74, 77, 84]
    })
else:
    df = None
    st.warning("⚠️ Carica un file o usa il dataset di esempio per iniziare.")

# -----------------------------
# Mostra dati se caricati
# -----------------------------
if df is not None:
    df = calcola_valore(df)

    st.subheader("📊 Anteprima Giocatori")
    st.dataframe(df.head())

    # Filtro per ruolo
    ruolo = st.selectbox("Filtra per ruolo", ["Tutti"] + sorted(df["Ruolo"].unique().tolist()))
    if ruolo != "Tutti":
        df = df[df["Ruolo"] == ruolo]

    # Top 10 per valore
    st.subheader("🏅 Top 10 per rapporto valore/prezzo")
    top10 = df.sort_values(by="Valore", ascending=False).head(10)
    st.dataframe(top10[["Nome", "Squadra", "Ruolo", "Prezzo", "Media", "Titolare", "Valore"]])

    fig = px.bar(top10, x="Nome", y="Valore", color="Ruolo", title="Top 10 Valore")
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Simulatore di squadra
    # -----------------------------
    st.subheader("🧮 Simulatore squadra ottimizzata")

    budget = st.number_input("Inserisci il tuo budget totale", min_value=1, value=100)
    slot_att = st.number_input("Numero attaccanti", min_value=0, value=3)
    slot_cen = st.number_input("Numero centrocampisti", min_value=0, value=4)
    slot_dif = st.number_input("Numero difensori", min_value=0, value=3)
    slot_por = st.number_input("Numero portieri", min_value=0, value=1)

    if st.button("Suggerisci squadra"):
        slot_ruoli = {"A": slot_att, "C": slot_cen, "D": slot_dif, "P": slot_por}
        squadra, budget_residuo = suggerisci_squadra(df, budget, slot_ruoli)
        st.success(f"✅ Squadra suggerita con budget residuo: {budget_residuo:.2f}")
        st.dataframe(squadra[["Nome", "Ruolo", "Prezzo", "Media", "Titolare", "Score"]])

    # -----------------------------
    # Predizione semplice
    # -----------------------------
    st.subheader("📈 Predizione performance")

    try:
        X = df[["Prezzo", "Titolare"]]
        y = df["Media"]
        model = LinearRegression().fit(X, y)
        df["Predizione"] = model.predict(X)

        fig2 = px.scatter(df, x="Media", y="Predizione", color="Ruolo", hover_data=["Nome"])
        fig2.add_shape(type="line", x0=df["Media"].min(), y0=df["Media"].min(),
                       x1=df["Media"].max(), y1=df["Media"].max(),
                       line=dict(color="Red", dash="dash"))
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.warning(f"Impossibile calcolare predizione: {e}")






