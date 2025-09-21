import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px

# --- FUNÇÃO PARA CARREGAR DADOS ---
@st.cache_data
def load_data(dataset_name):
    if dataset_name == 'Iris':
        data = load_iris()
        description = "Dataset Iris: 150 amostras, 4 features. Classes bem separadas."
    elif dataset_name == 'Wine':
        data = load_wine()
        description = "Dataset Wine: 178 amostras, 13 features. Mais complexo que o Iris."
    
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y, description

# --- FUNÇÃO PRINCIPAL DA ANÁLISE ---
def run_comparison_analysis(X, y, k_values, train_size, compare_scaling, weights_options, random_state):
    """
    Executa a análise comparativa de 'weights' no k-NN,
    considerando também o impacto da normalização, com MÚLTIPLAS REPETIÇÕES.
    """
    n_repetitions = 10 # Executar 10 vezes
    st.info(f"Executando cada configuração {n_repetitions} vezes para K = {k_values} com {int(train_size*100)}% de dados para treino. Isso pode levar um momento.")

    all_results = []
    scaling_options = ['Com Normalização', 'Sem Normalização'] if compare_scaling else ['Com Normalização']
    
    progress_bar = st.progress(0)
    total_iterations = n_repetitions * len(scaling_options) * len(weights_options) * len(k_values)
    current_iteration = 0

    # Loop de 10 repetições
    for i in range(n_repetitions):
        current_random_state = random_state + i # Usa uma seed diferente a cada vez

        # Loop para todas as combinações de parâmetros
        for scale_opt in scaling_options:
            for weight_opt in weights_options:
                for k in k_values:
                    current_iteration += 1
                    progress_bar.progress(current_iteration / total_iterations)

                    # Divisão dos dados
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, train_size=train_size, random_state=current_random_state, stratify=y
                    )

                    # Normalização condicional
                    if scale_opt == 'Com Normalização':
                        scaler = StandardScaler()
                        X_train_processed = scaler.fit_transform(X_train)
                        X_test_processed = scaler.transform(X_test)
                    else: # Sem Normalização
                        X_train_processed = X_train.values
                        X_test_processed = X_test.values

                    # Treinamento e Predição
                    try:
                        knn = KNeighborsClassifier(n_neighbors=k, weights=weight_opt)
                        knn.fit(X_train_processed, y_train)
                        y_pred = knn.predict(X_test_processed)
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        # Armazena o resultado bruto
                        all_results.append({
                            'K': k,
                            'Ponderação (Weights)': weight_opt,
                            'Normalização': scale_opt,
                            'Acurácia': accuracy
                        })
                    except Exception as e:
                        st.warning(f"Não foi possível executar para K={k} na repetição {i+1}: {e}")

    if not all_results:
        st.error("Nenhum resultado foi gerado. Verifique os parâmetros.")
        return

    # Processar os resultados para calcular média e desvio padrão
    results_df = pd.DataFrame(all_results)
    grouped = results_df.groupby(['K', 'Ponderação (Weights)', 'Normalização'])['Acurácia']
    stats_df = grouped.agg(['mean', 'std']).reset_index().fillna(0)
    stats_df.rename(columns={'mean': 'Acurácia Média', 'std': 'Desvio Padrão'}, inplace=True)

    # Exibir os resultados
    st.subheader("📊 Tabela de Resultados (Média e Desvio Padrão de 10 Execuções)")
    st.dataframe(stats_df.round(4), use_container_width=True)

    st.subheader("📈 Gráfico Comparativo de Desempenho (com Desvio Padrão)")
    
    fig = px.bar(
        stats_df,
        x='K',
        y='Acurácia Média',
        error_y='Desvio Padrão', # Adiciona barras de erro
        color='Ponderação (Weights)',
        barmode='group',
        facet_row='Normalização',
        title="Impacto da Ponderação ('uniform' vs 'distance') na Acurácia do k-NN",
        labels={'Acurácia Média': 'Acurácia Média do Modelo'},
        height=700
    )
    # Ajuste no eixo Y para acomodar a acurácia mais baixa do Wine sem normalização
    fig.update_yaxes(range=[0.4, 1.01]) 
    st.plotly_chart(fig, use_container_width=True)


# --- INTERFACE DO STREAMLIT ---
st.set_page_config(layout="wide")

st.title("🎯 Comparador de Ponderação no k-NN (Iris e Wine)")
st.markdown("Este aplicativo compara o desempenho do k-NN usando `weights='uniform'` vs. `weights='distance'`, mostrando também o impacto da normalização em diferentes datasets.")

# --- BARRA LATERAL COM OS CONTROLES ---
st.sidebar.header("⚙️ Parâmetros do Experimento")

dataset_choice = st.sidebar.selectbox(
    'Escolha o Dataset',
    ['Iris', 'Wine']
)

k_values = st.sidebar.multiselect(
    'Valores de K para Testar',
    options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15],
    default=[1, 3, 5, 7, 9],
    help="Valores de K ímpares são recomendados para evitar empates."
)

train_size = st.sidebar.slider(
    'Proporção de Dados para Treino',
    min_value=0.1, max_value=0.9, value=0.7, step=0.1
)

compare_scaling = st.sidebar.checkbox(
    'Analisar com e sem normalização',
    value=True,
    help="Se marcado, executa os testes duas vezes: uma com StandardScaler e outra sem."
)

weights_options = ['uniform', 'distance']
random_state = st.sidebar.number_input('Semente Aleatória (Random State)', value=42)

# --- EXECUÇÃO ---
# Carrega os dados com base na escolha da barra lateral
X, y, description = load_data(dataset_choice)
st.write(f"**Dataset Selecionado:** {dataset_choice}. {description}")

if st.button("🚀 Executar Análise Robusta (10 Repetições)"):
    if not k_values:
        st.warning("Por favor, selecione pelo menos um valor de K.")
    else:
        run_comparison_analysis(
            X=X,
            y=y,
            k_values=k_values,
            train_size=train_size,
            compare_scaling=compare_scaling,
            weights_options=weights_options,
            random_state=random_state
        )

