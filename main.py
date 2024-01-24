import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pickle
import numpy as np

# Load Skipgram embeddings
with open("embeddings_skipgram.pkl", 'rb') as f:
    skipgram_embeddings = pickle.load(f)

def get_word_vector(word, embs):
    try:
        return embs[word]
    except:
        return embs['<UNK>']

# Function to perform search using dot product for similarity
def search(query, top_k=10):
    query_embedding = get_word_vector(query.lower(), skipgram_embeddings)
    similarities = {}

    for vocab in skipgram_embeddings.keys():
        vocab_embedding = skipgram_embeddings[vocab]
        similarities[vocab] = np.dot(query_embedding, vocab_embedding)

    sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return sorted_results

# Create Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    html.H1("Search Engine"),
    dcc.Input(id='search-input', type='text', placeholder='Enter your query'),
    html.Button('Search', id='search-button', n_clicks=0),
    html.Div(id='search-results')
])

# Callback to update search results based on user input
@app.callback(
    Output('search-results', 'children'),
    [Input('search-button', 'n_clicks')],
    [dash.dependencies.State('search-input', 'value')]
)
def update_search_results(n_clicks, query):
    if query:
        results = search(query)
        result_items = [html.Li(f"{result[0]} - Dot Product: {result[1]:.4f}") for result in results]
        return html.Ul(result_items)
    else:
        return html.P("Enter a query and click the search button.")

if __name__ == '__main__':
    app.run(debug=True)

