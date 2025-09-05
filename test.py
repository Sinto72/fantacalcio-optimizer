# Asta Master - Streamlit app
# Filename Asta_Master_streamlit.py
# Description Strumento live di consulenza per l'asta del fantacalcio.
# Come usare
# 1. Metti il file Excel (es. giocatori.xlsx) nella stessa cartella oppure usa l'upload nell'interfaccia.
# 2. Installa dipendenze pip install -r requirements.txt
#    requirements.txt
#       pandas
#       numpy
#       streamlit
#       plotly
#       scikit-learn
#       requests
#       beautifulsoup4
# 3. Avvia streamlit run Asta_Master_streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import requests
from bs4 import BeautifulSoup
import time
import math

st.set_page_config(page_title=Asta Master, layout=wide)

# ---------------------- Helpers ----------------------
@st.cache_data
def load_data_from_excel(buffer)
    df = pd.read_excel(buffer)
    # Normalizza nomi colonne (se l'utente usa intestazioni leggermente diverse)
    cols = {c.upper() c for c in df.columns}
    # map known italian headings to our canonical ones
    mapping = {}
    for c in df.columns
        cu = c.strip().upper()
        if cu.startswith('NOM')
            mapping[c] = 'Nome'
        elif 'SQUAD' in cu
            mapping[c] = 'Squadra'
        elif 'RUOL' in cu or cu in ['A','C','D','P']
            mapping[c] = 'Ruolo'
        elif 'PREZ' in cu
            mapping[c] = 'Prezzo'
        elif 'MED' in cu
            mapping[c] = 'Media'
        elif 'TIT' in cu
            mapping[c] = 'Titolare'
        else
            # keep as is
            mapping[c] = c
    df = df.rename(columns=mapping)
    # Ensure required columns exist
    for col in ['Nome','Squadra','Ruolo','Prezzo','Media','Titolare']
        if col not in df.columns
            # create placeholder
            df[col] = np.nan
    # Convert types
    df['Prezzo'] = pd.to_numeric(df['Prezzo'], errors='coerce').fillna(0.0)
    df['Media'] = pd.to_numeric(df['Media'], errors='coerce').fillna(0.0)
    # Some excel have percent e.g. '74' o '0.74' - normalizziamo
    df['Titolare'] = pd.to_numeric(df['Titolare'], errors='coerce')
    df['Titolare'] = df['Titolare'].apply(lambda x x100 if (pd.notna(x) and x=1) else x)
    df['Titolare'] = df['Titolare'].fillna(0).astype(int)
    return df

# Valutazione valore rapporto Media  Prezzo (con smoothing per prezzo=0)
def compute_value_metrics(df)
    df = df.copy()
    df['Prezzo_smooth'] = df['Prezzo'].replace(0, 0.1)
    df['Valore'] = df['Media']  df['Prezzo_smooth']
    # Score combinato che pesa titolarita e media
    df['Score'] = df['Media']  (1 + df['Titolare']100)  (1 + np.log1p(df['Prezzo_smooth']))
    return df

# Simple knapsack-like greedy formation builder per ruolo
# input budget, desired_counts dict e.g. {'P'3,'D'8,'C'8,'A'6}
def suggest_team(df, budget, slots_per_role)
    df_sorted = df.copy()
    df_sorted = compute_value_metrics(df_sorted)
    suggestions = []
    remaining_budget = budget
    for role, slots in slots_per_role.items()
        candidates = df_sorted[df_sorted['Ruolo'] == role].sort_values(by='Score', ascending=False)
        picked = []
        for _, row in candidates.iterrows()
            if len(picked) = slots
                break
            if row['Prezzo'] = remaining_budget
                picked.append(row)
                remaining_budget -= row['Prezzo']
        # if can't fill with affordable ones, pick cheapest remaining
        if len(picked)  slots
            leftovers = candidates[~candidates['Nome'].isin([r['Nome'] for r in picked])].sort_values(by='Prezzo')
            for _, row in leftovers.iterrows()
                if len(picked) = slots
                    break
                if row['Prezzo'] = remaining_budget
                    picked.append(row)
                    remaining_budget -= row['Prezzo']
        suggestions.extend(picked)
    if len(suggestions) == 0
        return pd.DataFrame()
    return pd.DataFrame(suggestions)

# Mock function per interesse avversari - placeholder per scraping
@st.cache_data
def fetch_popularity_scores(df, query_sites=None)
    # Idealmente qui fai scraping dei siti di fantacalcio o Twitter per contare mention
    # In questa versione offline punteggio basato su Media e Prezzo (piu caropiu popolare)
    df = df.copy()
    df['Popolarita'] = (df['Media'] - df['Prezzo']10).rank(ascending=False)
    # Normalizza in 0-100
    df['Popolarita'] = 100  (df['Popolarita'].max() - df['Popolarita'])  (df['Popolarita'].max())
    return df

# Semplice predittore per punti futuri con regressione lineare usa Prezzo, Titolare e Media
@st.cache_data
def train_simple_predictor(df)
    dfc = df.copy()
    feats = ['Prezzo','Titolare','Media']
    X = dfc[feats].fillna(0)
    y = dfc['Media'].fillna(0)
    model = LinearRegression()
    model.fit(X, y)
    return model

# ---------------------- UI ----------------------
st.title("🚀 Asta Master - Consulente Live per l'Asta del Fantacalcio")
st.markdown('Strumento interattivo per suggerire acquisti durante la tua asta. Carica il file Excel con i giocatori o usa il dataset di esempio.')

col1, col2 = st.columns([2,1])
with col2
    st.header('Importa dati')
    uploaded = st.file_uploader('Carica il file Excel (giocatori.xlsx)', type=['xlsx','xls','csv'])
    use_example = st.button('Usa dataset d'esempio (se incluso)')

with col1
    if uploaded is not None
        df = load_data_from_excel(uploaded)
        st.success('File caricato con successo')
    else
        st.info('Nessun file caricato usa il button Usa dataset d'esempio oppure carica il tuo Excel.')
        # minimal fallback mostra istruzioni
        df = None

# If df not provided, try to load sample from embedded small table if user clicked
if df is None and use_example
    # Create a small sample from the chat data (a handful of rows) - user will likely upload real file
    sample = {
        'Nome' ['DAVID','MARTINEZ L.','MILINKOVIC-SAVIC V.','ZACCAGNI','THURAM'],
        'Squadra' ['JUV','INT','NAP','LAZ','INT'],
        'Ruolo' ['A','A','P','C','A'],
        'Prezzo' [146.55,174.2,13.05,74.8,163.8],
        'Media' [7.19,7.49,4.93,7.1,7.34],
        'Titolare' [85,89,20,89,89]
    }
    df = pd.DataFrame(sample)

if df is None
    st.stop()

# Preprocess
df = compute_value_metrics(df)

# Sidebar controls
st.sidebar.header('Parametri live')
budget = st.sidebar.number_input('Budget rimanente (fantamilioni)', value=300.0, min_value=0.0, step=1.0)
roles_default = {'P'3,'D'8,'C'8,'A'6}
st.sidebar.markdown('Slot desiderati per ruolo (totale consigliato)')
cols = st.sidebar.columns(4)
roles_counts = {}
for i, r in enumerate(['P','D','C','A'])
    roles_counts[r] = cols[i].number_input(r, value=roles_default[r], min_value=0, max_value=20)

# Livello di aggressività asta più alto = suggerimenti per spendere di più
aggressiveness = st.sidebar.slider('Aggressività asta', 0.0, 2.0, 1.0)

# Live search
st.sidebar.header('Ricerca rapida')
search_q = st.sidebar.text_input('Cerca giocatore per nome o squadra')

# Main columns
left, right = st.columns([2,1])

with left
    st.subheader('Top consigliati (valoreprezzo)')
    # compute Valore again
    df_display = df.copy()
    df_display['Valore'] = df_display['Media']  df_display['Prezzo'].replace(0, 0.1)
    # filter by search
    if search_q
        df_display = df_display[df_display['Nome'].str.contains(search_q, case=False, na=False)  df_display['Squadra'].str.contains(search_q, case=False, na=False)]
    role_filter = st.selectbox('Filtro ruolo (tutti)', options=['Tutti','A','C','D','P'])
    if role_filter != 'Tutti'
        df_display = df_display[df_display['Ruolo'] == role_filter]

    # Ranking
    top_by_value = df_display.sort_values(by='Valore', ascending=False).head(50)
    # color coding in dataframe via st.dataframe styler
    def color_valore(val)
        if val  1.5
            color = 'background-color #b8f2b8'  # greenish
        elif val  0.8
            color = 'background-color #fff3b0'  # yellow
        else
            color = 'background-color #ffd6d6'  # red
        return color
    st.dataframe(top_by_value[['Nome','Squadra','Ruolo','Prezzo','Media','Titolare','Valore']].style.applymap(color_valore, subset=['Valore']), height=400)

    st.markdown('---')
    st.subheader('Suggerimento squadra ottimizzata')
    suggested = suggest_team(df, budgetaggressiveness, roles_counts)
    if not suggested.empty
        st.write(f'Budget iniziale {budget.2f} — budget simulato (modificato dall'aggressività) {budgetaggressiveness.2f}')
        st.dataframe(suggested[['Nome','Squadra','Ruolo','Prezzo','Media','Titolare','Score']].sort_values(by='Score', ascending=False))
        st.write(f'Totale spesa simulata {suggested.Prezzo.sum().2f} — Giocatori selezionati {len(suggested)}')
    else
        st.info('Impossibile suggerire una squadra con i parametri forniti. Prova ad aumentare il budget o diminuire gli slot.')

    st.markdown('---')
    st.subheader('Predizioni punti (modello semplice)')
    model = train_simple_predictor(df)
    feats = df[['Prezzo','Titolare','Media']].fillna(0)
    preds = model.predict(feats)
    df['Predicted_Media'] = np.round(preds, 2)
    st.dataframe(df[['Nome','Squadra','Ruolo','Prezzo','Media','Predicted_Media']].sort_values(by='Predicted_Media', ascending=False).head(20))

with right
    st.subheader('Alert & Popolarità')
    df_pop = fetch_popularity_scores(df)
    top_pop = df_pop.sort_values(by='Popolarita', ascending=False).head(10)
    st.write('Giocatori più caldi (indice popolarità)')
    st.table(top_pop[['Nome','Squadra','Ruolo','Prezzo','Media','Popolarita']])

    st.markdown('---')
    st.subheader('Simulator asta competitiva (mock)')
    st.markdown('Qui lo strumento stima chi fra i tuoi avversari è più interessato a un giocatore usando Popolarità e prezzo.')
    candidate = st.selectbox('Scegli un giocatore per valutare interesse', options=df['Nome'].tolist())
    player_row = df[df['Nome'] == candidate].iloc[0]
    interest_score = float(player_row['Popolarita']) if 'Popolarita' in player_row.index else (player_row['Media']10 - player_row['Prezzo'])
    st.metric(label='Indice interesse stimato', value=f{interest_score.1f})
    st.write('Suggerimento offerta')
    # Esempio di suggerimento offerta prezzo raccomandato in funzione di budget e interesse
    def recommended_bid(price, interest, budget)
        base = price
        # se molto popolare aumenta la soglia
        multiplier = 1 + (interest100)  0.4
        # se budget poco, riduci
        budget_factor = min(1.0, budget100)
        return round(base  multiplier  budget_factor, 2)
    st.write(fPrezzo corrente stimato {player_row['Prezzo'].2f})
    st.write(fOfferta raccomandata {recommended_bid(player_row['Prezzo'], interest_score, budget).2f})

    st.markdown('---')
    st.subheader('Grafico valore vs prezzo')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Prezzo'], y=df['Media'], mode='markers', text=df['Nome'], marker=dict(size=8)))
    fig.update_layout(xaxis_title='Prezzo', yaxis_title='Media punti', height=350)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------- Utility buttons ----------------------
st.markdown('---')
colA, colB, colC = st.columns(3)
with colA
    if st.button('Esporta suggerimento come CSV')
        suggested = suggest_team(df, budgetaggressiveness, roles_counts)
        if suggested.empty
            st.warning('Nessun suggerimento da esportare')
        else
            csv = suggested.to_csv(index=False)
            st.download_button('Download CSV suggerimento', csv, file_name='suggerimento_asta.csv', mime='textcsv')
with colB
    if st.button('Aggiorna popolarità (tentativo scraping)')
        with st.spinner('Eseguo scraping rapido...')
            # Nota nella versione reale, inserire qui scraping di siti affidabili
            time.sleep(1)
            df = fetch_popularity_scores(df)
            st.success('Popolarità aggiornata (mock)')
with colC
    if st.button('Reset filtri')
        st.experimental_rerun()

# ---------------------- Footer ----------------------
st.markdown('---')
st.markdown('Note tecniche & prossimi step consigliati')
st.markdown('''
- Sostituisci la funzione `fetch_popularity_scores` con scraping reale (ad esempio dalle pagine di fantacalcio popolari o Twitter) per avere alert affidabili.
- Integra un motore di predizione più sofisticato (XGBoost, feature di calendario, infortuni, dati di rendimento per minuto).
- Per aggiornamenti live veloci, esegui lo script su un laptop con connessione e aggiorna la dashboard ogni 5-10 secondi usando `st.experimental_rerun()` o `st_autorefresh`.
- Per simulare gli interessi degli avversari, crea un micro-servizio che raccoglie le offerte fatte in asta e ne stima i pattern d'acquisto.
''')

st.markdown('Buona asta! Se vuoi, posso adattare il codice per')
st.markdown('- aggiungere scraping reale da siti italiani di fantacalcio
- migliorare il modello predittivo
- aggiungere notifiche sonore o via Telegram quando un giocatore diventa caldo')

