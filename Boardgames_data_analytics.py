#!/usr/bin/env python
# coding: utf-8

# # 🎲 Exploring Board Game Trends with BGG Data
# 
# ### 📌 Project overview
# 
# This notebook is part of a larger project that aims to analyze and visualize trends in the board game industry using data from [BoardGameGeek (BGG)](https://boardgamegeek.com), the largest online board game database. Our goal is to better understand how games evolve over time — in terms of mechanics, themes, complexity, popularity and player perception.
# 
# We will also leverage AI/NLP techniques to enrich the dataset with additional insights extracted from game descriptions, such as gameplay style, thematic tone, replayability and learnability.
# 
# Results of this data processing will be used to create interactive dashboards in **Tableau Public**, enabling users to explore:
# - The most popular and highly rated game mechanics and categories over time
# - Trends in player preferences and game design
# - Difficulty levels and replay value across genres
# - AI-generated clusters and classifications of game styles
# 
# The next step of this project is to develop a **user-facing application** that helps players:
# - Find the perfect game to play based on their preferences (e.g. player count, duration, difficulty, favorite mechanics)
# - Discover new games to purchase or try based on personalized recommendations
# - Get inspired by current trends and community favorites
# 
# This recommendation system will use the enriched dataset and AI insights to suggest games tailored to the user’s needs, mood, and context — whether for a cozy two-player evening or a chaotic party night.
# 
# ### 🎯 Objectives for this part
# 
# - Collect and clean structured data from the **BGG API**
# - Analyze key attributes such as ratings, mechanics, categories, complexity, and publication year
# - Use **NLP models** to extract hidden patterns from game descriptions
# - Enrich the dataset with AI-inferred metrics (style, replayability, learnability)
# - Prepare a Tableau-ready dataset for interactive visualization
# 
# ---
# 
# Let’s get started by loading the data and exploring what BGG has to offer!
# and exploring what BGG has to offer!
# 

# ## 🧰 Step 0: Creating virtual environment & importing required libraries

# In[5]:


''' code for creating venv:
conda create -n bgg_env python=3.10
conda activate bgg_env
pip install pandas numpy==1.26.4 boardgamegeek2 requests-cache==0.5.2
pip install transformers scikit-learn
pip install beautifulsoup4 requests
pip install seaborn
pip install statsmodels
pip install jupyter notebook
pip install ipykernel
pip install plotly
python -m ipykernel install --user --name bgg_env --display-name "BGG NLP Env"

jupyter notebook

'''


# In[24]:


from boardgamegeek import BGGClient
import pandas as pd
import numpy as np
import time
from datetime import datetime
import re
import unicodedata
import traceback
import requests
from bs4 import BeautifulSoup
from collections import defaultdict


# ## 🧹 Step 1: Loading and cleaning the dataset

# ### 1.1 Importing TOP 100 games data

# In[11]:


# Choosing TOP 100 games
def get_top_100_game_ids():
    url = "https://boardgamegeek.com/browse/boardgame"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    game_ids = []

    rows = soup.select("tr[id^='row_']")

    for row in rows:
        link = row.find("a", href=True)
        if link:
            href = link["href"]
            # href wygląda jak: /boardgame/174430/gloomhaven
            parts = href.split("/")
            if len(parts) >= 3 and parts[2].isdigit():
                game_id = int(parts[2])
                game_ids.append(game_id)

    return game_ids

top_100_ids = get_top_100_game_ids()
print(f"✅ Found {len(top_100_ids)} game IDs.")


# In[13]:


# Initialize client
bgg = BGGClient()

# Narrow down the set of games to import into our database 
chosen_games = top_100_ids


# In[15]:


# Storage for game data
games_data = []

# Function to extract all available ranks into a dictionary
def extract_ranks(game):
    ranks_dict = {}
    try:
        for r in game.ranks:
            # Use .name for key and .value as rank
            if r.value and str(r.value).isdigit():
                ranks_dict[r.name] = int(r.value)
    except Exception as e:
        print(f"❌ Error extracting ranks for {game.name}: {e}")
    return ranks_dict


# Main loop
for id in chosen_games:
    try:
        print(f">>>>> Fetching game ID: {id}")
        game = bgg.game(game_id=id)

        print(f"📦 Inspecting game: {game.name} (ID: {game.id})")
        
        if game is None:
            print(f"⚠️ Game returned None (ID: {game_id})")
            continue

        if getattr(game, "expansion", False):
            print(f"🔁 Skipping expansion: {game.name} (ID: {game.id})")
            continue

        if getattr(game, "rating_average", None) is None or getattr(game, "year", None) is None:
            print(f"⚠️ Incomplete game data: {game.name} (ID: {game.id}) — skipping")
            continue

        ranks = extract_ranks(game)

        games_data.append({
            "game_id": getattr(game, "id", None),
            "name": getattr(game, "name", None),
            "yearpublished": getattr(game, "year", None),
            "average_rating": getattr(game, "rating_average", None),
            "average_rating_num": getattr(game, "users_rated", None),

            "rank_boardgame": ranks.get("boardgame", None),
            "rank_strategygames": ranks.get("strategygames", None),
            "rank_familygames": ranks.get("familygames", None),
            "rank_partygames": ranks.get("partygames", None),
            "rank_wargames": ranks.get("wargames", None),
            "rank_thematic": ranks.get("thematic", None),
            "rank_abstracts": ranks.get("abstracts", None),
            "rank_cgs": ranks.get("cgs", None),

            "minplayers": getattr(game, "min_players", None),
            "maxplayers": getattr(game, "max_players", None),
            "playingtime": getattr(game, "playing_time", None),
            "minplaytime": getattr(game, "min_playing_time", None),
            "maxplaytime": getattr(game, "max_playing_time", None),
            "min_age": getattr(game, "min_age", None),

            "complexity": getattr(game, "rating_average_weight", None),
            "complexity_num": getattr(game, "rating_num_weights", None),
            "expansions": getattr(game, "expansions", []),

            "mechanics": getattr(game, "mechanics", []),
            "categories": getattr(game, "categories", []),
            "families": getattr(game, "families", []),

            "description": getattr(game, "description", ""),

            "owned": getattr(game, "users_owned", None),
            "wanted": getattr(game, "users_wanting", None),
            "designers": getattr(game, "designers", []),
            "publishers": getattr(game, "publishers", [])
        })

        time.sleep(1)  # avoid API rate limits

    except Exception as e:
        print(f"❌ Error fetching game (ID: {game_id})")
        print(f"Exception type: {type(e).__name__}")
        print("Details:")
        traceback.print_exc()


# Convert to DataFrame and save to CSV
df = pd.DataFrame(games_data)
print(f"\n✅ Successfully imported {len(df)} games into the dataset.")

today_str = datetime.today().strftime("%Y-%m-%d")
filename = f"bgg_games_{today_str}.csv"
df.to_csv(filename, index=False)
print(f"\n✅ File saved as: {filename}")


# In[3]:


# Quick overview of data types and missing values
df.info()


# ### 1.2 Cleaning the data & preparing them for export to Tableau

# In[20]:


# Remove duplicated games (based on 'game_id')
before = len(df)
df.drop_duplicates(subset="game_id", inplace=True)
after = len(df)

print(f"✅ Removed {before - after} duplicate games (based on 'game_id')")


# In[22]:


# Make sure rank do not contarins non-numeric data
rank_cols = [col for col in df.columns if col.startswith("rank_")]
df[rank_cols] = df[rank_cols].apply(pd.to_numeric, errors="coerce")
invalid_counts = df[rank_cols].isna().sum()
print("✅ Rank columns converted to numeric. Non-numeric (or missing) entries per column:")
for col, count in invalid_counts.items():
    print(f"  - {col}: {count} null values")


# In[24]:


# Romove columns with less than 5% of data coverage
threshold = 0.05
row_count = len(df)
cols_to_drop = [col for col in df.columns if df[col].notna().sum() / row_count < threshold]
df.drop(columns=cols_to_drop, inplace=True)
print(f"🧹 Dropped {len(cols_to_drop)} columns with <5% data coverage: {cols_to_drop}")
print(f"📊 Final dataset shape: {df.shape}")


# In[312]:


# Convert lists (e.g. mechanics, designers) to semicolon-separated strings
list_cols = ["mechanics", "categories", "families", "designers", "publishers", "expansions"]

for col in list_cols:
    df[col] = df[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)

print(f"✅ Converted list-based columns to strings: {', '.join(list_cols)}")


# In[28]:


# Clean 'description' text field: remove HTML tags and extra whitespace
def clean_description(text):
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Normalize Unicode (np. é → é), usuń kontrolne
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if unicodedata.category(c)[0] != "C")

    # Remove special quote marks, replace smart quotes
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

df["description"] = df["description"].apply(clean_description)
print("✅ Cleaned and sanitized 'description' column for safe CSV export.")


# In[30]:


# Export cleaned dataset to CSV with date in filename
today_str = datetime.today().strftime("%Y-%m-%d")
filename = f"bgg_games_cleaned_{today_str}.csv"

df.to_csv(filename, index=False, encoding="utf-8-sig")
print(f"💾 Dataset saved to file: {filename}")


# ## 📊 Step 2. Analysis in place

# In[26]:


import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from collections import  defaultdict, Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SizeValidator
from scipy.stats import linregress
import colorsys


# In[30]:


# Before we begin exploring the dataset in Python, we want to make sure list-like columns
# (which were previously joined as strings for CSV export) are now restored to proper Python lists.
# This will allow us to easily analyze and aggregate values like mechanics, categories, and designers.

#Starting with importing cleaned file - if you want to use existing data you can easily change file name whithout executing first part of the code
today_str = datetime.today().strftime("%Y-%m-%d")
filename = f"bgg_games_cleaned_{today_str}.csv"
df = pd.read_csv(filename, encoding="utf-8-sig")
print(f"✅ File read: {filename}")
print(df.shape)
df.head()

columns_to_convert_back = ["mechanics", "categories", "families", "designers", "publishers", "expansions"]
for col in columns_to_convert_back:
    df[col] = df[col].apply(lambda x: [s.strip() for s in x.split(",")] if isinstance(x, str) else [])
print("✅ Successfully converted stringified list columns back into proper Python lists.")


# In[32]:


# Tworzymy przedziały rankingu
df["rank_bucket"] = df["rank_boardgame"].apply(lambda r: (
    "Top 5" if r <= 5 else
    "Top 20" if r <= 20 else
    "Top 50" if r <= 50 else
    "Top 100"
))
# Kolorki
rank_colors = {
    "Top 5": "#d62728",     # czerwony
    "Top 20": "#ff7f0e",    # pomarańczowy
    "Top 50": "#2ca02c",    # zielony
    "Top 100": "#1f77b4"    # niebieski
}


# ### 2.1 ⏳Publication date of most popular games ⭐
# Are new games better rated and occupy the entire top of the ranking?

# In[35]:


# wersja statyczna wykresu
'''
# Tworzymy przedziały rankingu
df["rank_bucket"] = df["rank_boardgame"].apply(lambda r: (
    "Top 5" if r <= 5 else
    "Top 20" if r <= 20 else
    "Top 50" if r <= 50 else
    "Top 100"
))

# Wyznaczenie zakresu lat
#min_year = df["yearpublished"].min()
#max_year = df["yearpublished"].max()
#df[df["yearpublished"] == df["yearpublished"].min()][["name", "yearpublished"]]
df_recent = df[df["yearpublished"] >= 1990].copy()

# Korelacja Spearmana
corr, pval = spearmanr(df_recent["yearpublished"], df_recent["rank_boardgame"])
#print(corr)
corr_label = (
    r"$\bf{No\ significant\ correlation}$" + "\n"
    "between boardgame release date\n"
    f"and its BGG rank (Spearman r = {corr:.2f})"
)

# Wykres
plt.figure(figsize=(14, 8))
scatter = sns.scatterplot(
    data=df_recent,
    x="yearpublished",
    y="rank_boardgame",
    hue="rank_bucket",
    size="owned",
    sizes=(40, 300),
    alpha=0.75,
    palette="tab10"
)

plt.title("Ranking BGG vs. Year of Publication", fontsize=16, weight='bold')
plt.xlabel("Year of publication")
plt.ylabel("Ranking position")
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(rotation=45)
plt.gca().invert_yaxis()


# Przygotowanie legendy
handles, labels = scatter.get_legend_handles_labels()
custom_labels = []
custom_handles = []

# Info o korelacji jako pierwszy wiersz
custom_labels.append(corr_label)
custom_handles.append(plt.Line2D([], [], color='white', alpha=0))

# odstęp
custom_labels.append(" ")
custom_handles.append(plt.Line2D([], [], color='white', alpha=0))

# 2. sekcja: Ranking position
custom_labels.append("Ranking position")
custom_handles.append(plt.Line2D([], [], color='white', alpha=0))
for h, l in zip(handles, labels):
    if l.startswith("Top"):
        custom_labels.append(l)
        custom_handles.append(h)

# odstęp
custom_labels.append(" ")
custom_handles.append(plt.Line2D([], [], color='white', alpha=0))
        
# 3. sekcja: Popularity
custom_labels.append("Popularity (users owning the game)")
custom_handles.append(plt.Line2D([], [], color='white', alpha=0))

for h, l in zip(handles, labels):
    if l.isdigit():
        custom_labels.append(l)
        custom_handles.append(h)

# Finalna legenda
plt.legend(
    custom_handles,
    custom_labels,
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    title="KEY",
    borderpad=1.2
)

plt.tight_layout()
plt.show()
'''


# In[37]:


# Tworzymy przedziały rankingu
df["rank_bucket"] = df["rank_boardgame"].apply(lambda r: (
    "Top 5" if r <= 5 else
    "Top 20" if r <= 20 else
    "Top 50" if r <= 50 else
    "Top 100"
))


# Wyznaczenie zakresu lat
#min_year = df["yearpublished"].min()
#max_year = df["yearpublished"].max()
#df[df["yearpublished"] == df["yearpublished"].min()][["name", "yearpublished"]]
df_recent = df[df["yearpublished"] >= 1990].copy()


# Korelacja Spearmana
corr, _ = spearmanr(df_recent["yearpublished"], df_recent["rank_boardgame"])
#print(corr)


# Budujemy hover tekst
df_recent["hover_info"] = (
    "<b>" + df_recent["name"] + "</b><br>" +
    "Year: <b>" + df_recent["yearpublished"].astype(str) + "</b><br>" +
    "Rank: <b>" + df_recent["rank_boardgame"].astype(str) + "</b> (" + df_recent["rank_bucket"] + ")<br>" +
    "Average rating: <b>" + df_recent["average_rating"].round(2).astype(str) + "</b><br>" +
    "Number of ratings: " + df_recent["average_rating_num"].astype(str) + "<br>"
    "Owned by: " + df_recent["owned"].astype(str) + " users"
)

# Kolorki
rank_colors = {
    "Top 5": "#d62728",     # czerwony
    "Top 20": "#ff7f0e",    # pomarańczowy
    "Top 50": "#2ca02c",    # zielony
    "Top 100": "#1f77b4"    # niebieski
}


# 📈 Interaktywny wykres
fig1 = px.scatter(
    df_recent,
    x="yearpublished",
    y="rank_boardgame",
    color="rank_bucket",
    color_discrete_map=rank_colors,
    size="owned",
    size_max=20,
    custom_data=["hover_info"],
    hover_data=[],
    hover_name=None,
    labels={
        "yearpublished": "Year of publication",
        "rank_boardgame": "Ranking position",
        "rank_bucket": "Ranking",
        "owned": "Owned"
    },
    title="<b>BGG rank⭐ vs Year of publication⌛</b>",
    height=700
)


# Dodaj legendgroup i legendgrouptitle do grupy rankingowej
for trace in fig1.data:
    trace.legendgroup = "ranking"
    trace.legendgrouptitle = dict(text="<b>Ranking buckets</b>")


# Styl wykresu
for trace in fig1.data:
    if trace.mode == "markers" and hasattr(trace, "customdata") and trace.customdata is not None:
        #trace.name = "" << aktywacja tej linijki sprawia, że usuwamy tap bucketa obok tooltipa, ale znikają też labelki bucketów z legendy
        trace.hovertemplate = "%{customdata[0]}<extra></extra>"
        trace.hoverinfo = "skip"
        trace.showlegend = True
        trace.marker.opacity = 0.7
        trace.marker.line = dict(width=0.5, color='DarkSlateGrey')

fig1.update_layout(
    legend_title_text="",
    yaxis_autorange="reversed",  # bo rank 1 na górze
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(showgrid=True, gridcolor="lightgray"),
    yaxis=dict(showgrid=True, gridcolor="lightgray"),
    title_x=0,
    title_xanchor="left",
    title_font=dict(size=20, family="Arial", color="black"),
    hoverlabel=dict(bgcolor="white", font_size=13),
    font=dict(family="Arial", size=14),
    margin=dict(l=40, r=300, t=100, b=120),
)


# Przygotowanie legendy bąbelków
fig1.add_trace(go.Scatter(
    x=[None], y=[None],
    mode='lines',
    line=dict(width=0),
    name=" ",  # <- niewidzialny znak (ALT+0160 lub spacja)
    showlegend=True,
    legendgroup=None,
    hoverinfo="skip"
))
# Jeden symboliczny bąbelek do legendy
fig1.add_trace(go.Scatter(
    x=[None], y=[None],
    mode='markers',
    marker=dict(
        size=14,
        sizemode='area',
        color='gray',
        opacity=0.4,
        line=dict(width=0)
    ),
    name="= bubble size",
    legendgroup="popularity",
    legendgrouptitle=dict(
        text="<b>Popularity</b><br><span style='font-weight:normal'>How many users own the game?</span>"
    ),
    showlegend=True,
    hoverinfo="skip"
))
# Nieskuteczna próba pokazania realnej wilkości bąbelków w legendzie
'''size_max = 20
size_range = [df_recent["owned"].min(), df_recent["owned"].max()]
sizeref = 2. * max(size_range) / (size_max ** 2)
legend_sizes = df_recent["owned"].quantile([0.25, 0.5, 0.75, 1.0]).round().astype(int).unique()

for i, s in enumerate(legend_sizes):
    fig1.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            size=(s / sizeref) ** 0.5,
            sizemode='area',
            sizeref=sizeref,
            sizemin=2,
            color='gray',
            opacity=0.4,
            line=dict(width=0)
        ),
        name=f"{s:,} users",
        legendgroup="popularity",
        legendgrouptitle=dict(
            text="<b>Popularity</b><br><span style='font-weight:normal'>(owned by users)</span>"
        ) if i == 0 else None,
        showlegend=True,
        hoverinfo="skip"
    ))
'''

# Podtytuł
fig1.add_annotation(
    text="Are new games better rated and occupy the top of the ranking?",
    xref="paper", yref="paper",
    x=-0.09, y=1.08,
    xanchor="left",
    showarrow=False,
    font=dict(size=14, color="gray")
)


# ℹ️ Korelacja
fig1.add_annotation(
    text=f"<b>r = {corr:.2f}</b><br><span style='font-size:11px'>(Pearson correlation)</span>",
    xref="paper", yref="paper",
    x=0.01, y=0.99,
    showarrow=False,
    align="left",
    font=dict(size=12),
    bgcolor="white",
    bordercolor="lightgray",
    borderwidth=1
)
# Notatka o braku korelacji
fig1.add_annotation(
    text=(
        "<b>Note:</b> There is no significant correlation between a game's release date and its BGG rank "
        f"(Spearman r = {corr:.2f}).<br>"
    ),
    xref="paper", yref="paper",
    x=-0.09, y=-0.2,  # Pozycja poniżej wykresu
    xanchor="left",
    showarrow=False,
    align="left",
    font=dict(size=13, color="black"),
    bgcolor="white"
)


fig1.show()
fig1.write_html("bgg_rank_vs_year.html")


# ### 2.2 🎲 Most common mechanics

# In[39]:


# Budujemy listę wszystkich mechanik
all_mechanics = []
mechanic_to_titles = defaultdict(list)

for _, row in df.iterrows():
    for mech in row["mechanics"]:
        if isinstance(mech, str) and mech.strip():
            clean_mech = mech.strip()
            all_mechanics.append(clean_mech)
            mechanic_to_titles[clean_mech].append(row["name"])

# Zliczamy wystąpienia
def count_from_lists(df, column):
    all_items = []
    for entry in df[column].dropna():
        if isinstance(entry, list):
            all_items.extend([x.strip() for x in entry if isinstance(x, str)])
    return Counter(all_items)
    
mechanic_counts = Counter(all_mechanics)
print(f"🧮 # of unique mechanics: {len(mechanic_counts)}")
top_mechanics = mechanic_counts.most_common(15)

# 📊 Budujemy DataFrame do wykresu
plot_df = pd.DataFrame({
    "Mechanic": [x[0] for x in top_mechanics],
    "Count": [x[1] for x in top_mechanics],
}) 


plot_df["Percentage"] = round(plot_df["Count"] / len(df) * 100, 1)
plot_df["Number of games"] = plot_df["Count"]
plot_df["Examples"] = plot_df["Mechanic"].apply(lambda m: ", ".join(mechanic_to_titles[m][:4]))

# Tooltip HTML-style
plot_df["Tooltip"] = (
    "<b>Mechanic:</b> " + plot_df["Mechanic"] + "<br>" +
    "<b>Appears in:</b> " + plot_df["Number of games"].astype(str) + " games<br>" +
    "<b>Examples:</b> " + plot_df["Examples"]
)

# 📈 Plotly wykres
fig2 = px.bar(
    plot_df,
    x="Percentage",
    y="Mechanic",
    orientation="h",
    text="Percentage",
    hover_name="Mechanic",
        hover_data={
        "Tooltip": True,
        "Number of games": False,
        "Examples": False,
        "Count": False
    },
    title="<b>🎲 Most common (TOP15) mechanics in TOP100 BGG boardgames</b>",
    category_orders={
        "Mechanic": plot_df.sort_values("Percentage", ascending=False)["Mechanic"].tolist()
    },    
    color=plot_df["Percentage"],
    color_continuous_scale="emrld",
    height=800
)

# ✨ Stylizacja
fig2.update_traces(
    hovertemplate=plot_df["Tooltip"],
    textposition="outside",
    marker_line_width=0.5
)

fig2.update_layout(
    yaxis_title="Mechanic",
    xaxis_title="% of games",
    coloraxis_showscale=False,
    barmode="group",
    title_x=0,
    title_xanchor="left",
    title_font=dict(size=20, family="Arial", color="black"),
    hoverlabel=dict(bgcolor="white", font_size=13),
    uniformtext_minsize=8,
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(
        showline=True,
        linecolor='black',
        gridcolor='lightgray',
        zeroline=False
    ),
    yaxis=dict(
        showline=False
    )
)

fig2.show()


# ### 2.3 🎨 Most common categories

# In[43]:


import pandas as pd
import plotly.express as px
from collections import defaultdict, Counter

# Zakładamy, że df jest już wczytany i gotowy
# Budujemy listę wszystkich kategorii
all_categories = []
category_to_titles = defaultdict(list)

for _, row in df.iterrows():
    for cat in row["categories"]:
        if isinstance(cat, str) and cat.strip():
            clean_cat = cat.strip()
            all_categories.append(clean_cat)
            category_to_titles[clean_cat].append(row["name"])

# 🔢 Zliczamy wystąpienia
category_counts = Counter(all_categories)
print(f"🧮 # of unique categories: {len(category_counts)}")
top_categories = category_counts.most_common(15)

# 📊 Budujemy DataFrame do wykresu
plot_df = pd.DataFrame({
    "Category": [x[0] for x in top_categories],
    "Count": [x[1] for x in top_categories],
}) 

plot_df["Percentage"] = round(plot_df["Count"] / len(df) * 100, 1)
plot_df["Number of games"] = plot_df["Count"]
plot_df["Examples"] = plot_df["Category"].apply(lambda c: ", ".join(category_to_titles[c][:4]))

# Tooltip HTML-style
plot_df["Tooltip"] = (
    "<b>Category:</b> " + plot_df["Category"] + "<br>" +
    "<b>Appears in:</b> " + plot_df["Number of games"].astype(str) + " games<br>" +
    "<b>Examples:</b> " + plot_df["Examples"]
)

# 📈 Plotly wykres
fig3 = px.bar(
    plot_df,
    x="Percentage",
    y="Category",
    orientation="h",
    text="Percentage",
    hover_name="Category",
    hover_data={
        "Tooltip": True,
        "Number of games": False,
        "Examples": False,
        "Count": False
    },
    title="<b>🎨 Most common (TOP15) categories in TOP100 BGG boardgames</b>",
    category_orders={
        "Category": plot_df.sort_values("Percentage", ascending=False)["Category"].tolist()
    },
    color=plot_df["Percentage"],
    color_continuous_scale="emrld",
    height=800
)

# ✨ Stylizacja
fig3.update_traces(
    hovertemplate=plot_df["Tooltip"],
    textposition="outside",
    marker_line_width=0.5
)

fig3.update_layout(
    yaxis_title="Category",
    xaxis_title="% of games",
    coloraxis_showscale=False,
    barmode="group",
    title_x=0,
    title_xanchor="left",
    title_font=dict(size=20, family="Arial", color="black"),
    hoverlabel=dict(bgcolor="white", font_size=13),
    uniformtext_minsize=8,
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(
        showline=True,
        linecolor='black',
        gridcolor='lightgray',
        zeroline=False
    ),
    yaxis=dict(
        showline=False
    )
)

fig3.show()


# ### 2.4 Most common mechanics 🎲 inside categories 🎨
# What is best performing mechanics for given category?

# In[46]:


# Zliczanie
combo_counter = Counter()
category_totals = Counter()
mechanic_totals = Counter()
example_titles = {}

for _, row in df.iterrows():
    if isinstance(row["categories"], list) and isinstance(row["mechanics"], list):
        for cat in row["categories"]:
            if not isinstance(cat, str) or not cat.strip():
                continue
            category_totals[cat] += 1
            for mech in row["mechanics"]:
                if isinstance(mech, str) and mech.strip():
                    mechanic_totals[mech] += 1
                    combo_counter[(cat, mech)] += 1
                    key = (cat, mech)
                    if key not in example_titles:
                        example_titles[key] = []
                    if len(example_titles[key]) < 5:
                        example_titles[key].append(row["name"])

top_categories = [c for c, _ in category_totals.most_common(3)]
mechanic_counts = Counter()
for (cat, mech), count in combo_counter.items():
    if cat in top_categories:
        mechanic_counts[mech] += count
top_mechanics = [m for m, _ in mechanic_counts.most_common(5)]

# Funkcje pomocnicze
def percent_to_emerald(p):
    p = min(max(p, 0), 100) / 100
    return tuple(int(c * 255) for c in colorsys.hsv_to_rgb(0.4, 0.4, 1 - p))

def is_bright(rgb):
    r, g, b = rgb
    return (r*299 + g*587 + b*114) / 1000 > 125

# Rysowanie
fig4 = go.Figure()
x_labels = top_categories
y_labels = top_mechanics

for i, cat in enumerate(x_labels):
    for j, mech in enumerate(y_labels):
        key = (cat, mech)
        count = combo_counter.get(key, 0)
        total_cat = category_totals.get(cat, 1)
        total_mech = mechanic_totals.get(mech, 1)
        perc_cat = round(count / total_cat * 100, 1)
        perc_mech = round(count / total_mech * 100, 1)

        x0, x1 = i, i + 1
        y0, y1 = j, j + 1

        tooltip_cat = (
            f"<b>Category:</b> {cat}<br>"
            f"<b>Mechanic:</b> {mech}<br>"
            f"<b>Games with this combo:</b> {count}<br>"
            f"<b>Share (in category):</b> {perc_cat}%<br>"
            f"<b>Examples:</b> {', '.join(example_titles.get((cat, mech), []))}"
        )
        tooltip_mech = (
            f"<b>Mechanic:</b> {mech}<br>"
            f"<b>Category:</b> {cat}<br>"
            f"<b>Games with this combo:</b> {count}<br>"
            f"<b>Share (in mechanic):</b> {perc_mech}%<br>"
            f"<b>Examples:</b> {', '.join(example_titles.get((cat, mech), []))}"
        )

        # Dolny-lewy ↙ mechanika
        rgb_mech = percent_to_emerald(perc_mech)
        color_mech = f'rgb{rgb_mech}'
        text_color_mech = "black" if is_bright(rgb_mech) else "white"
        fig4.add_trace(go.Scatter(
            x=[x1, x1, x0],
            y=[y0, y1, y1],
            fill="toself",
            mode="lines",
            line=dict(color='gray'),
            fillcolor=color_mech,
            hoverinfo="skip",
            showlegend=False
        ))
        fig4.add_trace(go.Scatter(
            x=[(x0 + x1) / 2 + 0.2],
            y=[(y0 + y1) / 2 + 0.2],
            mode="text",
            text=[f"{perc_mech}%"],
            textfont=dict(color=text_color_mech, size=12),
            hoverinfo="text",
            hovertext=tooltip_mech,
            showlegend=False
        ))

        # Górny-prawy ↗ kategoria
        rgb_cat = percent_to_emerald(perc_cat)
        color_cat = f'rgb{rgb_cat}'
        text_color_cat = "black" if is_bright(rgb_cat) else "white"
        fig4.add_trace(go.Scatter(
            x=[x0, x0, x1],
            y=[y0, y1, y0],
            fill="toself",
            mode="lines",
            line=dict(color='gray'),
            fillcolor=color_cat,
            hoverinfo="skip",
            showlegend=False
        ))
        fig4.add_trace(go.Scatter(
            x=[(x0 + x1) / 2 - 0.2],
            y=[(y0 + y1) / 2 - 0.2],
            mode="text",
            text=[f"{perc_cat}%"],
            textfont=dict(color=text_color_cat, size=12),        
            hoverinfo="text",
            hovertext=tooltip_cat,
            showlegend=False
        ))

# Dodajemy opisy osi
fig4.add_annotation(
    text="<b>Mechanics</b>",
    xref="paper", yref="paper",
    x=-0.21, y=0.5,
    showarrow=False,
    font=dict(size=14),
    textangle=-90
)

fig4.add_annotation(
    text="<b>Category</b>",
    xref="paper", yref="paper",
    x=0.5, y=1.10,
    showarrow=False,
    font=dict(size=14)
)

# Stylizacja
fig4.update_layout(
    title=dict(
        text="<b>Most common mechanics🎲 inside top categories 🎨</b>",
        x=0, xanchor="left",
        font=dict(size=20)
    ),
    xaxis=dict(
        tickmode="array",
        tickvals=[i + 0.5 for i in range(len(x_labels))],
        ticktext=x_labels,
        side="top",
        range=[0, len(x_labels)],
        showgrid=False
    ),
    yaxis=dict(
        tickmode="array",
        tickvals=[j + 0.5 for j in range(len(y_labels))],
        ticktext=y_labels,
        autorange='reversed',
        range=[0, len(y_labels)],
        showgrid=False
    ),
    width=1000,
    height=800,
    plot_bgcolor="white",
    margin=dict(l=160, r=60, t=180, b=60)
)

fig4.add_annotation(
    text="What is best performing mechanics for given category?",
    xref="paper", yref="paper",
    x=-0.2067, y=1.138,
    showarrow=False,
    font=dict(size=14, color="gray"),
    xanchor="left"
)

fig4.show()


# ### 2.5 🧠 Complexity vs Rating ⭐
# Do BGG users prefer more complex games - and thus rate them higher?

# In[48]:


# 📊 Dane
df_scatter5 = df.copy()
df_scatter5 = df_scatter5[df_scatter5["complexity"].notna() & df_scatter5["average_rating"].notna()]

# 📈 Regresja liniowa
slope, intercept, r_value, p_value, std_err = linregress(df_scatter5["complexity"], df_scatter5["average_rating"])
df_scatter5["trendline"] = intercept + slope * df_scatter5["complexity"]

# 🧾 Tooltip
df_scatter5["tooltip"] = (
    "<b>" + df_scatter5["name"] + "</b><br>" +
    "Rank: <b>" + df_scatter5["rank_boardgame"].astype(str) + "</b> (" + df_scatter5["rank_bucket"] + ")<br>" +
    "Average rating: <b>" + df_scatter5["average_rating"].round(2).astype(str) + "</b><br>" +
    "Number of ratings: " + df_scatter5["average_rating_num"].round(0).astype(int).astype(str) + "<br>" +
    "Complexity: <b>" + df_scatter5["complexity"].round(2).astype(str) + "</b><br>" +
    "Number of complexity ratings:" + df_scatter5["complexity_num"].astype(str)
)

# 🎯 Wykres bazowy
fig5 = px.scatter(
    df_scatter5,
    x="complexity",
    y="average_rating",
    color="rank_bucket",
    size="average_rating_num",
    custom_data=["tooltip"],
    hover_data=[],
    hover_name=None,
    labels={
        "complexity": "Complexity (weight)",
        "average_rating": "Average user rating",
        "average_rating_num": "Number of ratings",
        "rank_bucket": "Rank bucket"
    },
    color_discrete_map=rank_colors,
    title="<b>Complexity🧠 vs Rating⭐ of BGG Top 100 Games</b>",
    height=700
)

# 🛠️ Stylizacja hoverów i markerów
for trace in fig5.data:
    if trace.mode == "markers":
        trace.hovertemplate = "%{customdata[0]}<extra></extra>"
        trace.marker.opacity = 0.7
        trace.marker.line = dict(width=0.5, color='DarkSlateGrey')

# Usuwamy stare wpisy z legendy
fig5.for_each_trace(
    lambda t: t.update(showlegend=False) if t.name in rank_colors else None
)

# ➕ Dodajemy nagłówki i legendy niestandardowe
fig5.add_trace(go.Scatter(x=[None], y=[None], mode="none", showlegend=True, name=" "))
fig5.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="lines",
    line=dict(width=0),
    showlegend=True,
    name="<b>Ranking buckets</b>",
    hoverinfo="skip"
))
for label, color in rank_colors.items():
    fig5.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=10, symbol="square", color=color),
        legendgroup="ranking",
        name=label,
        showlegend=True
    ))

# 📉 Trendline
fig5.add_trace(go.Scatter(x=[None], y=[None], mode="none", showlegend=True, name=" "))
fig5.add_trace(go.Scatter(
    x=df_scatter5["complexity"],
    y=df_scatter5["trendline"],
    mode="lines",
    name="<b>Trendline</b>",
    line=dict(color="black", dash="dash"),
    legendgroup="extra",
    showlegend=True
))

# 📈 Popularność (bubble size)
fig5.add_trace(go.Scatter(x=[None], y=[None], mode="none", showlegend=True, name=" "))
fig5.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="markers",
    marker=dict(size=10, color="lightgray"),
    legendgroup="extra",
    name="<b>Popularity</b><br><span style='font-weight:normal'>(bubble size = number of users rating)</span>",
    showlegend=True
))

# ℹ️ Korelacja
fig5.add_annotation(
    text=f"<b>r = {r_value:.2f}</b><br><span style='font-size:11px'>(Pearson correlation)</span>",
    xref="paper", yref="paper",
    x=0.01, y=0.99,
    showarrow=False,
    align="left",
    font=dict(size=12),
    bgcolor="white",
    bordercolor="lightgray",
    borderwidth=1
)

# 🎨 Styl
fig5.update_layout(
    legend_title_text="",
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(showgrid=True, gridcolor="lightgray"),
    yaxis=dict(showgrid=True, gridcolor="lightgray"),
    title_x=0,
    title_font=dict(size=20, family="Arial", color="black"),
    hoverlabel=dict(bgcolor="white", font_size=13),
    font=dict(family="Arial", size=14),
    margin=dict(l=40, r=260, t=90, b=50)
)

# Podtytuł
fig5.add_annotation(
    text="Do BGG users prefer more complex games - and thus rate them higher?",
    xref="paper", yref="paper",
    x=-0.08, y=1.06,
    xanchor="left",
    showarrow=False,
    font=dict(size=14, color="gray")
)

fig5.show()


# ### 2.6 Complexity🧠 vs Gameplay time 🕒 
# Is playtime strictly correlated with game complexity?

# In[51]:


# 📊 Dane
df_scatter6 = df.copy()
df_scatter6 = df_scatter6[df_scatter6["complexity"].notna() & df_scatter6["playingtime"].notna()]
df_scatter6 = df_scatter6[df_scatter6["playingtime"] < 720]


# 📈 Regresja
slope, intercept, r_value, _, _ = linregress(df_scatter6["complexity"], df_scatter6["playingtime"])
df_scatter6["trendline"] = intercept + slope * df_scatter6["complexity"]

# 🧾 Tooltip
df_scatter6["tooltip"] = (
    "<b>" + df_scatter6["name"] + "</b><br>" +
    "Playing time: <b>" + df_scatter6["playingtime"].astype(str) + " min</b><br>" +
    "Complexity: <b>" + df_scatter6["complexity"].round(2).astype(str) + "</b><br>" +
    "Number of complexity ratings: " + df_scatter6["complexity_num"].astype(str)
)

# 🎯 Wykres
fig6 = px.scatter(
    df_scatter6,
    x="complexity",
    y="playingtime",
    color="rank_bucket",
    size="average_rating_num",
    custom_data=["tooltip"],
    hover_data=[],
    hover_name=None,
    labels={
        "complexity": "Complexity (weight)",
        "playingtime": "Playing time (minutes)",
        "average_rating_num": "Number of ratings"
    },
    color_discrete_map=rank_colors,
    title="<b>Complexity🧠 vs Playing Time🕒 of BGG Top 100 Games</b>",
    height=700
)


# 🛠️ Stylizacja hoverów i markerów
for trace in fig6.data:
    if trace.mode == "markers":
        trace.hovertemplate = "%{customdata[0]}<extra></extra>"
        trace.marker.opacity = 0.7
        trace.marker.line = dict(width=0.5, color='DarkSlateGrey')

# Usuwamy stare wpisy z legendy
fig6.for_each_trace(
    lambda t: t.update(showlegend=False) if t.name in rank_colors else None
)

# ➕ Dodajemy nagłówki i legendy niestandardowe
fig6.add_trace(go.Scatter(x=[None], y=[None], mode="none", showlegend=True, name=" "))
fig6.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="lines",
    line=dict(width=0),
    showlegend=True,
    name="<b>Ranking buckets</b>",
    hoverinfo="skip"
))
for label, color in rank_colors.items():
    fig6.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=10, symbol="square", color=color),
        legendgroup="ranking",
        name=label,
        showlegend=True
    ))

# 📉 Trendline
fig6.add_trace(go.Scatter(x=[None], y=[None], mode="none", showlegend=True, name=" "))
fig6.add_trace(go.Scatter(
    x=df_scatter6["complexity"],
    y=df_scatter6["trendline"],
    mode="lines",
    name="<b>Trendline</b>",
    line=dict(color="black", dash="dash"),
    legendgroup="extra",
    showlegend=True
))

# 📈 Popularność (bubble size)
fig6.add_trace(go.Scatter(x=[None], y=[None], mode="none", showlegend=True, name=" "))
fig6.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="markers",
    marker=dict(size=10, color="lightgray"),
    legendgroup="extra",
    name="<b>Popularity</b><br><span style='font-weight:normal'>(bubble size = number of users rating)</span>",
    showlegend=True
))

# ℹ️ Korelacja
fig6.add_annotation(
    text=f"<b>r = {r_value:.2f}</b><br><span style='font-size:11px'>(Pearson correlation)</span>",
    xref="paper", yref="paper",
    x=0.01, y=0.99,
    showarrow=False,
    align="left",
    font=dict(size=12),
    bgcolor="white",
    bordercolor="lightgray",
    borderwidth=1
)

# 🎨 Styl
fig6.update_layout(
    legend_title_text="",
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(showgrid=True, gridcolor="lightgray"),
    yaxis=dict(showgrid=True, gridcolor="lightgray"),
    title_x=0,
    title_font=dict(size=20, family="Arial", color="black"),
    hoverlabel=dict(bgcolor="white", font_size=13),
    font=dict(family="Arial", size=14),
    margin=dict(l=40, r=260, t=90, b=50)
)

# Podtytuł
fig6.add_annotation(
    text="Is playtime strictly correlated with game complexity?",
    xref="paper", yref="paper",
    x=-0.09, y=1.06,
    xanchor="left",
    showarrow=False,
    font=dict(size=14, color="gray")
)

fig6.show()


# ### 2.7 🧠 Complexity vs Number of Expansions 🚀
# Do complex games get more expansions, or less?

# In[54]:


# 📊 Dane
df_scatter7 = df.copy()
df_scatter7["num_expansions"] = df_scatter7["expansions"].apply(lambda x: len(x) if isinstance(x, list) else 0)
df_scatter7 = df_scatter7[df_scatter7["complexity"].notna() & df_scatter7["num_expansions"].notna()]
df_scatter7["expansion_names"] = df_scatter7["expansions"].apply(
    lambda lst: ", ".join(lst[:10]) + ("..." if len(lst) > 10 else "") if isinstance(lst, list) else "None"
)


# 📈 Regresja liniowa
slope, intercept, r_value, _, _ = linregress(df_scatter7["complexity"], df_scatter7["num_expansions"])
df_scatter7["trendline"] = intercept + slope * df_scatter7["complexity"]

# 🧾 Tooltip
df_scatter7["tooltip"] = (
    "<b>" + df_scatter7["name"] + "</b><br>" +
    "# of expansions: <b>" + df_scatter7["num_expansions"].astype(str) + "</b><br>" +
    #"Expansions: <b>" + df_scatter7["expansion_names"] + "</b><br>" +
    "Complexity: <b>" + df_scatter7["complexity"].round(2).astype(str) + "</b><br>" +
    "Number of complexity ratings:" + df_scatter7["complexity_num"].astype(str)
)

# 🎯 Wykres bazowy
fig7 = px.scatter(
    df_scatter7,
    x="complexity",
    y="num_expansions",
    color="rank_bucket",
    size="average_rating_num",
    custom_data=["tooltip"],
    hover_data=[],
    hover_name=None,
    labels={
        "complexity": "Complexity (weight)",
        "num_expansions": "Number of expansions",
        "average_rating_num": "Number of ratings"
    },
    color_discrete_map=rank_colors,
    title="<b>Complexity🧠 vs Number of Expansions🚀 in BGG Top 100</b>",
    height=700
)

# 🛠️ Stylizacja hoverów i markerów
for trace in fig7.data:
    if trace.mode == "markers":
        trace.hovertemplate = "%{customdata[0]}<extra></extra>"
        trace.marker.opacity = 0.7
        trace.marker.line = dict(width=0.5, color='DarkSlateGrey')

# Usuwamy stare wpisy z legendy
fig7.for_each_trace(
    lambda t: t.update(showlegend=False) if t.name in rank_colors else None
)

# ➕ Dodajemy nagłówki i legendy niestandardowe
fig7.add_trace(go.Scatter(x=[None], y=[None], mode="none", showlegend=True, name=" "))
fig7.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="lines",
    line=dict(width=0),
    showlegend=True,
    name="<b>Ranking buckets</b>",
    hoverinfo="skip"
))
for label, color in rank_colors.items():
    fig7.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=10, symbol="square", color=color),
        legendgroup="ranking",
        name=label,
        showlegend=True
    ))

# 📉 Trendline
fig7.add_trace(go.Scatter(x=[None], y=[None], mode="none", showlegend=True, name=" "))
fig7.add_trace(go.Scatter(
    x=df_scatter7["complexity"],
    y=df_scatter7["trendline"],
    mode="lines",
    name="<b>Trendline</b>",
    line=dict(color="black", dash="dash"),
    legendgroup="extra",
    showlegend=True
))

# 📈 Popularność (bubble size)
fig7.add_trace(go.Scatter(x=[None], y=[None], mode="none", showlegend=True, name=" "))
fig7.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="markers",
    marker=dict(size=10, color="lightgray"),
    legendgroup="extra",
    name="<b>Popularity</b><br><span style='font-weight:normal'>(bubble size = number of users rating)</span>",
    showlegend=True
))

# ℹ️ Korelacja
fig7.add_annotation(
    text=f"<b>r = {r_value:.2f}</b><br><span style='font-size:11px'>(Pearson correlation)</span>",
    xref="paper", yref="paper",
    x=0.01, y=0.99,
    showarrow=False,
    align="left",
    font=dict(size=12),
    bgcolor="white",
    bordercolor="lightgray",
    borderwidth=1
)

# 🎨 Styl
fig7.update_layout(
    legend_title_text="",
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(showgrid=True, gridcolor="lightgray"),
    yaxis=dict(showgrid=True, gridcolor="lightgray"),
    title_x=0,
    title_font=dict(size=20, family="Arial", color="black"),
    hoverlabel=dict(bgcolor="white", font_size=13),
    font=dict(family="Arial", size=14),
    margin=dict(l=40, r=260, t=90, b=50)
)

# Podtytuł
fig7.add_annotation(
    text="Do complex games get more expansions, or less?",
    xref="paper", yref="paper",
    x=-0.08, y=1.06,
    xanchor="left",
    showarrow=False,
    font=dict(size=14, color="gray")
)

fig7.show()


# ### 2.8 🎲 Mechanics vs Number of Expansions 🚀
# Are some mechanics more prone to be expanded by new game additions?

# In[56]:


# 🧠 Przygotowanie danych
df["num_expansions"] = df["expansions"].apply(lambda x: len(x) if isinstance(x, list) else 0)
df_mech = df.explode("mechanics")
df_mech = df_mech[df_mech["mechanics"].notna() & df_mech["mechanics"].apply(lambda x: isinstance(x, str) and x.strip())]

# 🎯 Agregacja
grouped = df_mech.groupby("mechanics").agg(
    AvgExpansions=("num_expansions", "mean"),
    GameCount=("name", "count"),
    Examples=("name", lambda x: ", ".join(x.dropna().unique()[:4]))
).reset_index()

# 🔝 Top 15 mechanik z największą liczbą gier
plot_df = grouped.sort_values("GameCount", ascending=False).head(15).copy()
plot_df.rename(columns={"mechanics": "Mechanic"}, inplace=True)
plot_df["AvgExpansions"] = plot_df["AvgExpansions"].round(2)

# 📎 Tooltip
plot_df["Tooltip"] = (
    "Mechanic: <b>" + plot_df["Mechanic"] + "</b><br>" +
    "Avg # of expansions: <b>" + plot_df["AvgExpansions"].astype(str) + "</b><br>" +
    "Games using this mechanic: <b>" + plot_df["GameCount"].astype(str) + "</b><br>" +
    "Examples: <b>" + plot_df["Examples"] + "</b>"
)

# 📈 Plotly wykres
fig8 = px.bar(
    plot_df,
    x="AvgExpansions",
    y="Mechanic",
    orientation="h",
    text="AvgExpansions",
    hover_name="Mechanic",
    hover_data={
        "Tooltip": True,
        "GameCount": False,
        "Examples": False
    },
    title="<b>🎲 Mechanics vs Number of Expansions 🚀</b>",
    category_orders={
        "Mechanic": plot_df.sort_values("AvgExpansions", ascending=False)["Mechanic"].tolist()
    },    
    color=plot_df["AvgExpansions"],
    color_continuous_scale="emrld",
    height=800
)

# ✨ Stylizacja
fig8.update_traces(
    hovertemplate=plot_df["Tooltip"],
    textposition="outside",
    marker_line_width=0.5
)

fig8.update_layout(
    yaxis_title="Mechanic",
    xaxis_title="Average # of expansions",
    coloraxis_showscale=False,
    barmode="group",
    title_x=0,
    title_xanchor="left",
    title_font=dict(size=20, family="Arial", color="black"),
    hoverlabel=dict(bgcolor="white", font_size=13),
    uniformtext_minsize=8,
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(
        showline=True,
        linecolor='black',
        gridcolor='lightgray',
        zeroline=False
    ),
    yaxis=dict(
        showline=False
    )
)


# Podtytuł
fig8.add_annotation(
    text="Are some mechanics more prone to be expanded by new game additions?",
    xref="paper", yref="paper",
    x=-0.35, y=1.06,
    xanchor="left",
    showarrow=False,
    font=dict(size=14, color="gray")
)


fig8.show()


# ### 2.9 Complexity🧠 and Number of Players👥 vs Popularity (Owned games🛒)
How complexity and (average) number of players affect probability of game purchase?
# 

# In[60]:


# 📊 Dane
df_bubble = df.copy()
df_bubble = df_bubble[df_bubble["complexity"].notna() & df_bubble["owned"].notna()]
df_bubble["num_expansions"] = df_bubble["expansions"].apply(lambda x: len(x) if isinstance(x, list) else 0)


# Średnia liczba graczy
df_bubble["average_players"] = (df_bubble["minplayers"] + df_bubble["maxplayers"]) / 2
df_bubble = df_bubble[df_bubble["average_players"].notna()]

# 🧾 Tooltip
df_bubble["tooltip"] = (
    "<b>" + df_bubble["name"] + "</b><br>" +
    "Owned by: <b>" + df_bubble["owned"].astype(int).astype(str) + "</b> users<br>" +
    "Complexity: <b>" + df_bubble["complexity"].round(2).astype(str) + "</b><br>" +
    "Avg. Players: <b>" + df_bubble["average_players"].round(2).astype(str) + "</b>"
)

# 📈 Wykres
fig9 = px.scatter(
    df_bubble,
    x="complexity",
    y="average_players",
    size="owned",
    color="rank_bucket",
    custom_data=["tooltip"],
    hover_data=[],
    hover_name=None,
    labels={
        "complexity": "Complexity (weight)",
        "average_players": "Average number of players",
        "owned": "Number of users owning the game",
        "rank_bucket": "Rank bucket"
    },
    color_discrete_map=rank_colors,
    title="<b>Complexity🧠 and Number of Players👥 vs Popularity (Owned games🛒)</b>",
    height=700
)

# ✨ Stylizacja hoverów i markerów
for trace in fig9.data:
    if trace.mode == "markers":
        trace.hovertemplate = "%{customdata[0]}<extra></extra>"
        trace.marker.opacity = 0.7
        trace.marker.line = dict(width=0.5, color='DarkSlateGrey')


fig9.for_each_trace(
    lambda t: t.update(showlegend=False) if t.name in rank_colors else None
)

# ➕ Legendy ręczne
fig9.add_trace(go.Scatter(x=[None], y=[None], mode="none", showlegend=True, name=" "))
fig9.add_trace(go.Scatter(
    x=[None], y=[None], mode="lines", line=dict(width=0), showlegend=True,
    name="<b>Ranking buckets</b>", hoverinfo="skip"
))
for label, color in rank_colors.items():
    fig9.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=10, symbol="square", color=color),
        legendgroup="ranking",
        name=label,
        showlegend=True
    ))

# 📈 Popularność (bubble size)
fig9.add_trace(go.Scatter(x=[None], y=[None], mode="none", showlegend=True, name=" "))
fig9.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="markers",
    marker=dict(size=10, color="lightgray"),
    legendgroup="extra",
    name="<b>Popularity</b><br><span style='font-weight:normal'>(bubble size = users owning the game)</span>",
    showlegend=True
))

# 🎨 Styl
fig9.update_layout(
    legend_title_text="",
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(showgrid=True, gridcolor="lightgray"),
    yaxis=dict(showgrid=True, gridcolor="lightgray"),
    title_x=0,
    title_font=dict(size=20, family="Arial", color="black"),
    hoverlabel=dict(bgcolor="white", font_size=13),
    font=dict(family="Arial", size=14),
    margin=dict(l=40, r=260, t=90, b=50)
)

# Podtytuł
fig9.add_annotation(
    text="How complexity and (average) number of players affect probability of game purchase?",
    xref="paper", yref="paper",
    x=-0.08, y=1.06,
    xanchor="left",
    showarrow=False,
    font=dict(size=14, color="gray")
)

fig9.show()


# In[71]:


# 📊 Dane
df_scatter10 = df.copy()
df_scatter10 = df_scatter10[df_scatter10["complexity"].notna() & df_scatter10["owned"].notna()]

# 📈 Regresja liniowa
slope, intercept, r_value, p_value, std_err = linregress(df_scatter10["complexity"], df_scatter10["owned"])
df_scatter10["trendline"] = intercept + slope * df_scatter10["complexity"]

# 🧾 Tooltip
df_scatter10["tooltip"] = (
    "<b>" + df_scatter10["name"] + "</b><br>" +
    "Rank: <b>" + df_scatter10["rank_boardgame"].astype(str) + "</b> (" +df_scatter10["rank_bucket"] + ")<br>" +
    "Average rating: <b>" + df_scatter10["average_rating"].round(2).astype(str) + "</b><br>" +
    "Number of ratings: " + df_scatter10["average_rating_num"].round(0).astype(int).astype(str) + "<br>" +
    "Complexity: <b>" + df_scatter10["complexity"].round(2).astype(str) + "</b><br>" +
    "Number of complexity ratings: " + df_scatter10["complexity_num"].astype(str) + "<br>" +
    "Owned by: <b>" + df_scatter10["owned"].astype(str)  + "users</b>"
)

# 🎯 Wykres
fig10 = px.scatter(
    df_scatter10,
    x="complexity",
    y="owned",
    color="rank_bucket",
    size="average_rating_num",
    custom_data=["tooltip"],
    hover_data=[],
    hover_name=None,
    labels={
        "complexity": "Complexity (weight)",
        "owned": "Number of players owning the game",
        "average_rating_num": "Number of ratings",
        "rank_bucket": "Rank bucket"
    },
    color_discrete_map=rank_colors,
    title="<b>Complexity🧠 vs Popularity📈 of BGG Top 100 Games</b>",
    height=700
)

# 🛠️ Stylizacja hoverów i markerów
for trace in fig10.data:
    if trace.mode == "markers":
        trace.hovertemplate = "%{customdata[0]}<extra></extra>"
        trace.marker.opacity = 0.7
        trace.marker.line = dict(width=0.5, color='DarkSlateGrey')

# Usuwamy stare wpisy z legendy
fig10.for_each_trace(
    lambda t: t.update(showlegend=False) if t.name in rank_colors else None
)

# ➕ Dodajemy nagłówki i legendy niestandardowe
fig10.add_trace(go.Scatter(x=[None], y=[None], mode="none", showlegend=True, name=" "))
fig10.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="lines",
    line=dict(width=0),
    showlegend=True,
    name="<b>Ranking buckets</b>",
    hoverinfo="skip"
))
for label, color in rank_colors.items():
    fig10.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=10, symbol="square", color=color),
        legendgroup="ranking",
        name=label,
        showlegend=True
    ))

# 📉 Trendline
fig10.add_trace(go.Scatter(x=[None], y=[None], mode="none", showlegend=True, name=" "))
fig10.add_trace(go.Scatter(
    x=df_scatter10["complexity"],
    y=df_scatter10["trendline"],
    mode="lines",
    name="<b>Trendline</b>",
    line=dict(color="black", dash="dash"),
    legendgroup="extra",
    showlegend=True
))

# 📈 Popularność (bubble size)
fig10.add_trace(go.Scatter(x=[None], y=[None], mode="none", showlegend=True, name=" "))
fig10.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="markers",
    marker=dict(size=10, color="lightgray"),
    legendgroup="extra",
    name="<b>Popularity</b><br><span style='font-weight:normal'>(bubble size = number of ratings)</span>",
    showlegend=True
))

# ℹ️ Korelacja
fig10.add_annotation(
    text=f"<b>r = {r_value:.2f}</b><br><span style='font-size:11px'>(Pearson correlation)</span>",
    xref="paper", yref="paper",
    x=0.01, y=0.99,
    showarrow=False,
    align="left",
    font=dict(size=12),
    bgcolor="white",
    bordercolor="lightgray",
    borderwidth=1
)

# 🎨 Styl
fig10.update_layout(
    legend_title_text="",
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(showgrid=True, gridcolor="lightgray"),
    yaxis=dict(showgrid=True, gridcolor="lightgray"),
    title_x=0,
    title_font=dict(size=20, family="Arial", color="black"),
    hoverlabel=dict(bgcolor="white", font_size=13),
    font=dict(family="Arial", size=14),
    margin=dict(l=40, r=260, t=90, b=50)
)

# Podtytuł
fig10.add_annotation(
    text="Are more complex games owned by more players?",
    xref="paper", yref="paper",
    x=-0.092, y=1.06,
    xanchor="left",
    showarrow=False,
    font=dict(size=14, color="gray")
)

fig10.show()


# In[73]:


# 📊 Dane
df_scatter11 = df.copy()
df_scatter11 = df_scatter11[
    df_scatter11["minplayers"].notna() &
    df_scatter11["maxplayers"].notna() &
    df_scatter11["owned"].notna()
]
# Średnia liczba graczy
df_scatter11["average_players"] = (df_scatter11["minplayers"] + df_scatter11["maxplayers"]) / 2


# 📈 Regresja liniowa
slope, intercept, r_value, p_value, std_err = linregress(df_scatter11["average_players"], df_scatter11["owned"])
df_scatter11["trendline"] = intercept + slope * df_scatter11["average_players"]

# 🧾 Tooltip
df_scatter11["tooltip"] = (
    "<b>" + df_scatter11["name"] + "</b><br>" +
    "Rank: <b>" + df_scatter11["rank_boardgame"].astype(str) + "</b> (" + df_scatter11["rank_bucket"] + ")<br>" +
    "Average rating: <b>" + df_scatter11["average_rating"].round(2).astype(str) + "</b><br>" +
    "Number of ratings: " + df_scatter11["average_rating_num"].round(0).astype(int).astype(str) + "<br>" +
    "Players: <b>" + df_scatter11["minplayers"].astype(str) + "–" + df_scatter11["maxplayers"].astype(str) + "</b> (avg: " + df_scatter11["average_players"].round(1).astype(str) + ")<br>" +
    "Owned by: <b>" + df_scatter11["owned"].astype(str)  + "users</b>"
)

# 🎯 Wykres
fig11 = px.scatter(
    df_scatter11,
    x="average_players",
    y="owned",
    color="rank_bucket",
    size="average_rating_num",
    custom_data=["tooltip"],
    hover_data=[],
    hover_name=None,
    labels={
        "average_players": "Average number of players",
        "owned": "Number of players owning the game",
        "average_rating_num": "Number of ratings",
        "rank_bucket": "Rank bucket"
    },
    color_discrete_map=rank_colors,
    title="<b>Avg. Number of Players 👥 vs Popularity📈 of BGG Top 100 Games</b>",
    height=700
)

# 🛠️ Stylizacja hoverów i markerów
for trace in fig11.data:
    if trace.mode == "markers":
        trace.hovertemplate = "%{customdata[0]}<extra></extra>"
        trace.marker.opacity = 0.7
        trace.marker.line = dict(width=0.5, color='DarkSlateGrey')

# Usuwamy stare wpisy z legendy
fig11.for_each_trace(
    lambda t: t.update(showlegend=False) if t.name in rank_colors else None
)

# ➕ Dodajemy nagłówki i legendy niestandardowe
fig11.add_trace(go.Scatter(x=[None], y=[None], mode="none", showlegend=True, name=" "))
fig11.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="lines",
    line=dict(width=0),
    showlegend=True,
    name="<b>Ranking buckets</b>",
    hoverinfo="skip"
))
for label, color in rank_colors.items():
    fig11.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=10, symbol="square", color=color),
        legendgroup="ranking",
        name=label,
        showlegend=True
    ))

# 📉 Trendline (poprawka – sortujemy i agregujemy)
trend_df = df_scatter11[["average_players", "trendline"]].copy()
trend_df = trend_df.sort_values("average_players")
trend_df = trend_df.drop_duplicates(subset="average_players")

fig11.add_trace(go.Scatter(
    x=trend_df["average_players"],
    y=trend_df["trendline"],
    mode="lines",
    name="<b>Trendline</b>",
    line=dict(color="black", dash="dash"),
    legendgroup="extra",
    showlegend=True
))


# 📈 Popularność (bubble size)
fig11.add_trace(go.Scatter(x=[None], y=[None], mode="none", showlegend=True, name=" "))
fig11.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="markers",
    marker=dict(size=10, color="lightgray"),
    legendgroup="extra",
    name="<b>Popularity</b><br><span style='font-weight:normal'>(bubble size = number of ratings)</span>",
    showlegend=True
))

# ℹ️ Korelacja
fig11.add_annotation(
    text=f"<b>r = {r_value:.2f}</b><br><span style='font-size:11px'>(Pearson correlation)</span>",
    xref="paper", yref="paper",
    x=0.01, y=0.99,
    showarrow=False,
    align="left",
    font=dict(size=12),
    bgcolor="white",
    bordercolor="lightgray",
    borderwidth=1
)

# 🎨 Styl
fig11.update_layout(
    legend_title_text="",
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(showgrid=True, gridcolor="lightgray"),
    yaxis=dict(showgrid=True, gridcolor="lightgray"),
    title_x=0,
    title_font=dict(size=20, family="Arial", color="black"),
    hoverlabel=dict(bgcolor="white", font_size=13),
    font=dict(family="Arial", size=14),
    margin=dict(l=40, r=260, t=90, b=50)
)

# Podtytuł
fig11.add_annotation(
    text="Are games for more players owned by more users?",
    xref="paper", yref="paper",
    x=-0.094, y=1.06,
    xanchor="left",
    showarrow=False,
    font=dict(size=14, color="gray")
)

fig11.show()


# In[75]:


# ## 📊 Step 2.10 Analysis in place - summary 📌

# ### 1. BGG Rank ⭐ vs. Year of Publication ⌛  
# **Observation:** Most top-ranked games have been published after 2010, with very few older titles (pre-2000) present. 
# However, looking at the 2010–2015 period, there's only a weak correlation between publication date and rank. 
# The 2015–2020 range appears to be the most popular, likely due to the sheer volume of board games released in that time and the time it takes for a game to build reputation and get played.  
# **Conclusion:** Modern board games dominate the BGG Top 100, but that might be because the market expanded significantly after 2010, giving newer titles more opportunity to shine and be discovered.

# ---

# ### 3. 🎨 Most Common (TOP15) Categories in TOP100 BGG Board Games  
# **Top Categories:** "Economic", "Fantasy", "Science Fiction", "Fighting", and "Adventure" lead the list.  
# **Conclusion:** Fantasy and science fiction categories offer immersive experiences and new worlds to explore — making them more likely to gain popularity among players.

# ### 2. 🎲 Most Common (TOP15) Mechanics in TOP100 BGG Board Games  
# **Top Mechanics:** "Solo/Solitaire Game", "Hand Management", "Variable Player Powers", "Variable Set-up", and "Open Drafting" are the most frequent.  
# **Conclusion:** Mechanics requiring strategic resource management, dynamic game setup and player interaction are highly valued by the BGG community. 
# Solo modes are also common among top games, reflecting the popularity of solo-friendly designs.

# ---

# ### 7. 🎲 Mechanics vs. 🚀 Number of Expansions  
# **Observation:** The mechanics most associated with expansions include: "Scenario/Mission/Campaign Game", "Cooperative Game", "Solo/Solitaire Game", "Variable Player Powers", and "Dice Rolling".  
# **Conclusion:** Scenario-based games lend themselves well to expansions aiming to increase replayability. 
# Additionally, expansions often help introduce or refine new modes like solo or cooperative gameplay.

# ### 6. 🧠 Complexity vs. 🚀 Number of Expansions in BGG Top 100  
# **Observation:** No clear correlation (r = 0.09); games of varying complexity may have many or few expansions.  
# **Conclusion:** Expansions are likely driven more by popularity and commercial success — often supported by a big or loyal fanbase — rather than game complexity alone.

# ---

# ### 5. 🧠 Complexity vs. 🕒 Playtime of BGG Top 100 Games  
# **Observation:** A strong correlation exists (r = 0.66) — more complex games tend to have longer playtimes.  
# **Conclusion:** Complexity often comes hand-in-hand with depth and extended gameplay.

# ### 8. 🧠 Complexity vs. 👥 Number of Players  
# **Observation:** No strong correlation. Most games are designed for 2–4 players regardless of complexity.  
# **Conclusion:** Designers tend to stick to the optimal 2–4 player range, as it balances accessibility and strategic engagement.

# ### 10. 👥 Avg. Number of Players vs. 📉 Popularity of BGG Top 100 Games  
# **Observation:** No significant relationship between average number of players and BGG rank.  
# **Conclusion:** Player count does not heavily influence a game’s popularity; other factors like mechanics, theme, and gameplay depth have a stronger impact.

# ### 4. 🧠 Complexity vs. ⭐ Rating of BGG Top 100 Games  
# **Observation:** There's a slight positive correlation (r = 0.34) between complexity and rating.  
# **Conclusion:** BGG users tend to appreciate more complex games, as the community skews toward experienced and enthusiast players.

# ### 9. 🧠 Complexity vs. 📉 Popularity of BGG Top 100  
# **Observation:** There's a slight negative correlation (r = –0.31) — simpler games are more broadly popular.  
# **Conclusion:** While complex games dominate the Top 100 in terms of ratings, simplicity allows certain titles to reach a wider audience. 
# Simpler games tend to have clearer consensus on top picks, while complex games show more variation in rankings — possibly due to longer engagement times and appeal to more specialized audiences.

# ---

# ## 📌 Summary  
# The BGG Top 100 is dominated by modern, strategic, and complex games.  

# - **Complexity** is positively correlated with **longer playtime** and **higher user ratings**, but not necessarily with broader popularity — simpler games can reach a wider audience.  
# - **Most popular mechanics** emphasize **decision-making**, **resource management**, and **asymmetrical play**.  
# - In terms of **categories**, storytelling and world-building elements are key: **fantasy**, **science fiction**, and **adventure** rank highest.  
# - **Expansions** are most frequent in games featuring **scenarios**, **solo modes**, and **cooperative mechanics**, supporting additional content and replayability.


# In[ 76 ]:

import os
import base64
#from weasyprint import HTML

# 📅 Folder z datą
today_str = datetime.today().strftime("%Y-%m-%d")
output_dir = today_str
os.makedirs(output_dir, exist_ok=True)

# 📊 Figury
figures = [
    ("BGG_rank_vs_Year", fig1),
    ("Top15_Mechanics", fig2),
    ("Top15_Categories", fig3),
    ("Complexity_vs_Rating", fig5),
    ("Complexity_vs_PlayingTime", fig6),
    ("Complexity_vs_Expansions", fig7),
    ("Mechanics_vs_Expansions", fig8),
    ("Complexity_and_Players_vs_Popularity", fig9),
    ("Complexity_vs_Popularity", fig10),
    ("Players_vs_Popularity", fig11)
]

# Definicje opisów dla każdej figury
descriptions = {
    "BGG_rank_vs_Year": {
        "title": "BGG Rank ⭐ vs. Year of Publication ⌛",
        "description": ("<b>Observation:</b> Most top-ranked games have been published after 2010, with very few older titles (pre-2000) present. "
                        "However, during the 2010–2015 period, there's only a weak correlation between publication date and rank. "
                        "The 2015–2020 range appears to be the most popular, likely due to the high volume of releases and time needed for a game to build reputation.<br/>"
                        "<b>Conclusion:</b> Modern board games dominate the BGG Top 100, possibly because the market expanded significantly after 2010, giving newer titles more opportunity to shine.")
    },
    "Top15_Mechanics": {
        "title": "🎲 Most Common (TOP15) Mechanics in TOP100 BGG Board Games",
        "description": ("<b>Top Mechanics:</b> \"Solo/Solitaire Game\", \"Hand Management\", \"Variable Player Powers\", \"Variable Set-up\", and \"Open Drafting\" are the most frequent.<br/>"
                        "<b>Conclusion:</b> Mechanics requiring strategic resource management, dynamic game setup, and player interaction are highly valued by the BGG community. "
                        "Solo modes are also common among top games, reflecting the popularity of solo-friendly designs.")
    },
    "Top15_Categories": {
        "title": "🎨 Most Common (TOP15) Categories in TOP100 BGG Board Games",
        "description": ("<b>Top Categories:</b> \"Economic\", \"Fantasy\", \"Science Fiction\", \"Fighting\", and \"Adventure\" lead the list.<br/>"
                        "<b>Conclusion:</b> Fantasy and science fiction categories offer immersive experiences and new worlds to explore, making them more likely to gain popularity among players.")
    },
    "Complexity_vs_Rating": {
        "title": "Complexity 🧠 vs. Rating ⭐ of BGG Top 100 Games",
        "description": ("<b>Observation:</b> There's a slight positive correlation (r = 0.34) between complexity and rating.<br/>"
                        "<b>Conclusion:</b> BGG users tend to appreciate more complex games, as the community skews toward experienced and enthusiast players.")
    },
    "Complexity_vs_PlayingTime": {
        "title": "Complexity 🧠 vs. Playtime 🕒 of BGG Top 100 Games",
        "description": ("<b>Observation:</b> A strong correlation exists (r = 0.66) — more complex games tend to have longer playtimes.<br/>"
                        "<b>Conclusion:</b> Complexity often comes hand-in-hand with depth and extended gameplay.")
    },
    "Complexity_vs_Expansions": {
        "title": "Complexity 🧠 vs. Number of Expansions 🚀 of BGG Top 100 Games",
        "description": ("<b>Observation:</b> No clear correlation exists (r = 0.09); games of varying complexity may have many or few expansions.<br/>"
                        "<b>Conclusion:</b> Expansions appear to be driven more by popularity and commercial success—often supported by a strong or loyal fanbase—than by game complexity alone.")
    },
    "Mechanics_vs_Expansions": {
        "title": "Mechanics 🎲 vs. Number of Expansions 🚀",
        "description": ("<b>Observation:</b> The mechanics most associated with expansions include: \"Scenario/Mission/Campaign Game\", \"Cooperative Game\", "
                        "\"Solo/Solitaire Game\", \"Variable Player Powers\", and \"Dice Rolling\".<br/>"
                        "<b>Conclusion:</b> Scenario-based games are well-suited for expansions to boost replayability, and expansions may help introduce or refine new gameplay modes like solo or cooperative play.")
    },
    "Complexity_and_Players_vs_Popularity": {
        "title": "Complexity 🧠 & Number of Players 👥 vs. Popularity 📉 ",
        "description": ("<b>Observation:</b> The relationship between complexity, number of players, and popularity is not strongly defined, indicating that factors beyond these metrics influence game popularity.<br/>"
                        "<b>Conclusion:</b> While game complexity and player count are important, other elements such as mechanics and theme also play a significant role in a game's success.")
    },
    "Complexity_vs_Popularity": {
        "title": "Complexity 🧠 vs. Popularity 📉 of BGG Top 100 Games ",
        "description": ("<b>Observation:</b> There's a slight negative correlation (r = –0.31) — simpler games tend to be more broadly popular.<br/>"
                        "<b>Conclusion:</b> Although complex games dominate the Top 100 in ratings, simplicity helps some titles reach a wider audience. "
                        "Simpler games often have a clearer consensus on top picks, whereas complex games display more ranking variation, possibly due to longer engagement times.")
    },
    "Players_vs_Popularity": {
        "title": "Avg. Number of Players 👥 vs. Popularity 📉 of BGG Top 100 Games ",
        "description": ("<b>Observation:</b> There is no significant relationship between the average number of players and a game's BGG rank.<br/>"
                        "<b>Conclusion:</b> The number of players does not heavily influence a game's popularity; other factors such as mechanics, theme, and gameplay depth are more influential.")
    }
}

# Budowanie części HTML z obrazami i opisami
html_parts = []

for fig_key, fig in figures:
    # Przygotuj nazwę pliku
    filename = f"{fig_key.replace(' ', '_').lower()}.png"
    path = f"{filename}"
    
    # Eksportuj obraz do PNG
    pio.write_image(fig, path, width=1200, height=800)
    
    # Konwersja do base64
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    
    # Pobierz tytuł i opis z definicji
    title = descriptions.get(fig_key, {}).get("title", fig_key)
    description = descriptions.get(fig_key, {}).get("description", "")
    
    # Jeśli tekst zawiera zarówno Observation i Conclusion – rozbij na dwa paragrafy
    if "<b>Conclusion:</b>" in description:
        obs, concl = description.split("<b>Conclusion:</b>", 1)
    
        obs_html = f'<p style="font-size:14px; line-height:1.6; color:#333; margin-bottom:18px;"><b>Observation:</b>{obs.strip().replace("<b>Observation:</b>", "")}</p>'
        concl_html = f'<p style="font-size:14px; line-height:1.6; color:#333;"><b>Conclusion:</b>{concl.strip()}</p>'
    else:
        # fallback – cały tekst jako jeden paragraf
        obs_html = f'<p style="font-size:14px; line-height:1.6; color:#333;">{description.strip()}</p>'
        concl_html = ""
    
    html_block = f"""
    <div style="max-width:900px; margin:40px auto;">
        <h2 style="text-align:left; font-size:20px; color:#222; margin-bottom:12px;">{title}</h2>
        <img src="data:image/png;base64,{encoded}" width="800px" style="display:block; margin-bottom:10px;" />
        {obs_html}
        {concl_html}
    </div>
    """



    html_parts.append(html_block)

# Składa raport HTML
final_html = f"""
<html>
<head>
    <meta charset="utf-8">
    <title>BGG Visualizations Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #fff;
            color: #333;
        }}
        h1 {{
            color: #111;
            text-align: center;
        }}
        h2 {{
            color: #444;
            margin-top: 60px;
        }}
        p {{
            font-size: 14px;
            line-height: 1.6;
        }}
        hr {{
            border: 0;
            height: 1px;
            background: #ccc;
            margin-top: 40px;
            margin-bottom: 40px;
        }}
        img {{
            border: 1px solid #ccc;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <h1>📊 BGG Visualization Report</h1>
    {''.join(html_parts)}
</body>
</html>
"""

# 💾 Zapisz HTML do pliku
html_filename = os.path.join(output_dir, f"BGG_report_with_images_{today_str}.html")
with open(html_filename, "w", encoding="utf-8") as f:
    f.write(final_html)

print(f"✅ HTML zapisany: {html_filename}")

'''
# 📄 ➡️ PDF – konwersja HTML do PDF przy użyciu WeasyPrint
try:
    from weasyprint import HTML

    pdf_filename = os.path.join(output_dir, f"BGG_report_with_images_{today_str}.pdf")
    HTML(html_filename).write_pdf(pdf_filename)

    print(f"✅ PDF wygenerowany: {pdf_filename}")
except Exception as e:
    print("⚠️ PDF nie został wygenerowany (WeasyPrint).")
    print(e)
'''












# In[ ]:





# In[ ]:




