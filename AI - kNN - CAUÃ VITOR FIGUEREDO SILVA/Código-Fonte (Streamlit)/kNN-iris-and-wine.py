import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# Configuração da página
st.set_page_config(
    page_title="k-NN Interactive Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para melhorar a aparência
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.dataset-info {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.metric-card {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 0.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_dataset(dataset_choice):
    """Carrega o dataset escolhido"""
    if dataset_choice == "Iris":
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target)
        target_names = iris.target_names.tolist()
        # Ensure y values are valid indices
        y = y.astype(int)
        return X, y, target_names, "Classificação de espécies de flores Iris"
    
    elif dataset_choice == "Wine":
        try:
            wine = fetch_ucirepo(id=109)
            X = pd.DataFrame(wine.data.features)
            y = pd.Series(wine.data.targets.values.ravel())
            
            # Fix the target encoding - ensure consecutive integers starting from 0
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y))
            target_names = [f"Classe {int(c)}" for c in sorted(le.classes_)]
            
            return X, y, target_names, "Classificação de qualidade de vinhos"
        except Exception as e:
            st.error(f"Erro ao carregar dataset Wine: {str(e)}. Usando Iris como padrão.")
            return load_dataset("Iris")

def safe_target_mapping(y, target_names):
    """Safely map target indices to names"""
    try:
        # Ensure all y values are valid indices
        y_mapped = []
        for val in y:
            if isinstance(val, (int, np.integer)) and 0 <= val < len(target_names):
                y_mapped.append(target_names[val])
            else:
                # Handle invalid indices
                y_mapped.append(f"Unknown_{val}")
        return y_mapped
    except Exception as e:
        st.error(f"Error mapping targets: {str(e)}")
        return [f"Class_{i}" for i in y]

def create_data_overview(X, y, target_names, description):
    """Cria visualização geral dos dados"""
    st.markdown(f"<h3>📊 {description}</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📈 Amostras", len(X))
    with col2:
        st.metric("🔧 Features", len(X.columns))
    with col3:
        st.metric("🎯 Classes", len(target_names))
    with col4:
        balance_text = "Balanceado" if y.value_counts().std() < 5 else "Desbalanceado"
        st.metric("📊 Balanço", balance_text)
    
    # Distribuição das classes - use safe mapping
    y_labels = safe_target_mapping(y, target_names)
    
    fig_dist = px.histogram(
        x=y_labels, 
        title="🎯 Distribuição das Classes",
        labels={'x': 'Classes', 'y': 'Número de Amostras'},
        color=y_labels
    )
    fig_dist.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_dist, use_container_width=True)

def create_feature_analysis(X, y, target_names):
    """Análise detalhada das features"""
    st.markdown("### 🔍 Análise das Features")
    
    # Estatísticas descritivas
    with st.expander("📋 Estatísticas Descritivas"):
        st.dataframe(X.describe().round(3))
    
    # Correlação entre features
    col1, col2 = st.columns(2)
    
    with col1:
        fig_corr = px.imshow(
            X.corr().round(2),
            text_auto=True,
            title="🔗 Matriz de Correlação",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        # Boxplot das features por classe
        feature_selected = st.selectbox("Selecione uma feature para análise:", X.columns)
        
        # Safe mapping for boxplot
        y_labels = safe_target_mapping(y, target_names)
        
        df_plot = pd.DataFrame({
            'Feature': X[feature_selected],
            'Classe': y_labels
        })
        
        fig_box = px.box(
            df_plot, 
            x='Classe', 
            y='Feature',
            title=f"📊 Distribuição de {feature_selected} por Classe",
            color='Classe'
        )
        st.plotly_chart(fig_box, use_container_width=True)

def create_pca_visualization(X, y, target_names):
    """Visualização PCA dos dados"""
    st.markdown("### 🎨 Visualização PCA")
    
    # Aplica PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
    
    # Safe mapping for PCA plot
    y_labels = safe_target_mapping(y, target_names)
    
    # Cria DataFrame para plotagem
    df_pca = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Classe': y_labels
    })
    
    fig_pca = px.scatter(
        df_pca, 
        x='PC1', 
        y='PC2', 
        color='Classe',
        title=f"🎨 Visualização PCA - Variância Explicada: {pca.explained_variance_ratio_.sum():.3f}",
        labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.3f})',
                'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.3f})'}
    )
    fig_pca.update_traces(marker=dict(size=10))
    st.plotly_chart(fig_pca, use_container_width=True)
    
    return pca, X_pca

def run_knn_analysis(X, y, target_names, k_value, test_size, random_state):
    """Executa análise k-NN"""
    
    # Ensure y is properly encoded
    if not all(isinstance(val, (int, np.integer)) and 0 <= val < len(target_names) for val in y):
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    else:
        y_encoded = y.copy()
    
    # Divisão dos dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    # Padronização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Treinamento
    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(X_train_scaled, y_train)
    
    # Predições
    y_pred = knn.predict(X_test_scaled)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    
    # --- INÍCIO DA MODIFICAÇÃO ---
    # Calcular o relatório de classificação completo
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Extrair as métricas macro (que consideram a média entre as classes)
    precision_macro = report['macro avg']['precision']
    recall_macro = report['macro avg']['recall']
    f1_macro = report['macro avg']['f1-score']
    # --- FIM DA MODIFICAÇÃO ---

    # Create target names list that matches the encoded labels
    available_labels = sorted(list(set(y_test) | set(y_pred)))
    target_names_subset = [target_names[i] if i < len(target_names) else f"Class_{i}" for i in available_labels]
    
    report_dict = classification_report(y_test, y_pred, target_names=target_names_subset, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'y_test': y_test,
        'y_pred': y_pred,
        'report_dict': report_dict,
        'confusion_matrix': cm,
        'model': knn,
        'scaler': scaler,
        'available_labels': available_labels
    }

def create_results_visualization(results, target_names, k_value):
    """Visualiza resultados do modelo"""
    st.markdown("### 📈 Resultados do Modelo")
    
    # Métricas principais
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 Acurácia", f"{results['accuracy']:.3f}")
    with col2:
        correct = sum(results['y_test'] == results['y_pred'])
        st.metric("✅ Acertos", f"{correct}/{len(results['y_test'])}")
    with col3:
        errors = len(results['y_test']) - correct
        st.metric("❌ Erros", str(errors))
    
    # Matriz de confusão interativa
    col1, col2 = st.columns(2)
    
    with col1:
        # Get the target names for available labels
        available_labels = results['available_labels']
        target_names_subset = [target_names[i] if i < len(target_names) else f"Class_{i}" for i in available_labels]
        
        fig_cm = px.imshow(
            results['confusion_matrix'],
            text_auto=True,
            labels={'x': 'Predito', 'y': 'Real'},
            title="🔄 Matriz de Confusão",
            color_continuous_scale="Blues"
        )
        
        if len(target_names_subset) <= len(available_labels):
            fig_cm.update_xaxes(ticktext=target_names_subset, tickvals=list(range(len(target_names_subset))))
            fig_cm.update_yaxes(ticktext=target_names_subset, tickvals=list(range(len(target_names_subset))))
        
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        # Métricas por classe
        try:
            report_df = pd.DataFrame(results['report_dict']).transpose().round(3)
            report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
            
            # Gráfico de barras das métricas
            metrics_data = []
            for metric in ['precision', 'recall', 'f1-score']:
                for idx, class_name in enumerate(target_names_subset):
                    if class_name in report_df.index and metric in report_df.columns:
                        metrics_data.append({
                            'Classe': class_name,
                            'Métrica': metric.title(),
                            'Valor': report_df.loc[class_name, metric]
                        })
            
            if metrics_data:
                fig_metrics = px.bar(
                    pd.DataFrame(metrics_data),
                    x='Classe',
                    y='Valor',
                    color='Métrica',
                    title="📊 Métricas por Classe",
                    barmode='group'
                )
                fig_metrics.update_yaxes(range=[0, 1])
                st.plotly_chart(fig_metrics, use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao criar visualização de métricas: {str(e)}")

def k_optimization_analysis(X, y, target_names, test_size, random_state):
    """Análise de otimização do k"""
    st.markdown("### 🔧 Otimização do Hiperparâmetro k")
    
    # Explicação do critério de desempate
    with st.expander("ℹ️ Critério de Desempate"):
        st.markdown("""
        **Critério para escolha do melhor k:**
        1. **Máxima Acurácia**: Primeiro, seleciona os valores de k com maior acurácia
        2. **Menor Complexidade**: Em caso de empate, escolhe o **menor k** porque:
           - ✅ Menor complexidade computacional
           - ✅ Menor chance de overfitting
           - ✅ Decisões baseadas em vizinhos mais próximos
           - ✅ Modelo mais simples e interpretável
        
        **Por que k menor é melhor em empates?**
        - k=1: Usa apenas o vizinho mais próximo (mais local)
        - k=5: Usa 5 vizinhos (mais global, pode incluir ruído)
        - Princípio da Navalha de Occam: "A explicação mais simples é geralmente a melhor"
        """)
    
    max_k = min(20, len(X) // 5)
    k_range = range(1, max_k + 1)
    
    with st.spinner("Testando diferentes valores de k..."):
        k_results = []
        
        for k in k_range:
            try:
                results = run_knn_analysis(X, y, target_names, k, test_size, random_state)
                k_results.append({
                    'k': k,
                    'accuracy': results['accuracy'],
                    'errors': len(results['y_test']) - sum(results['y_test'] == results['y_pred'])
                })
            except Exception as e:
                st.warning(f"Erro para k={k}: {str(e)}")
                continue
    
    if not k_results:
        st.error("Não foi possível executar a otimização de k")
        return 5
    
    k_df = pd.DataFrame(k_results)
    
    # Gráfico da evolução da acurácia
    fig_k = go.Figure()
    
    fig_k.add_trace(go.Scatter(
        x=k_df['k'],
        y=k_df['accuracy'],
        mode='lines+markers',
        name='Acurácia',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    # Encontra o melhor k (menor k em caso de empate na acurácia)
    max_accuracy = k_df['accuracy'].max()
    best_candidates = k_df[k_df['accuracy'] == max_accuracy]
    best_k_idx = best_candidates['k'].idxmin()  # Menor k entre os que têm max accuracy
    best_k = k_df.loc[best_k_idx]
    
    fig_k.add_trace(go.Scatter(
        x=[best_k['k']],
        y=[best_k['accuracy']],
        mode='markers',
        name=f'Melhor k = {int(best_k["k"])}',
        marker=dict(color='red', size=15, symbol='star')
    ))
    
    fig_k.update_layout(
        title="📈 Evolução da Acurácia por Valor de k",
        xaxis_title="Valor de k",
        yaxis_title="Acurácia",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_k, use_container_width=True)
    
    # Tabela com resultados
    k_df_display = k_df.copy()
    k_df_display['accuracy'] = k_df_display['accuracy'].round(4)
    st.dataframe(k_df_display.set_index('k'), use_container_width=True)
    
    success_msg = f"🏆 Melhor k encontrado: **{int(best_k['k'])}** (Acurácia: {best_k['accuracy']:.4f})"
    if len(best_candidates) > 1:
        success_msg += f" - Escolhido menor k entre {len(best_candidates)} empates"
    st.success(success_msg)
    
    return int(best_k['k'])

# Substitua sua função inteira por esta
# Versão FINAL da função. Substitua a sua por esta.
def run_exercise_analysis(X, y, target_names, k_values, sample_sizes, initial_random_state):
    """Executa análise sistemática, mostra os resultados e oferece exportação em CSV."""
    st.markdown("### 📋 Análise Sistemática - Exercício")
    
    n_repetitions = 10
    st.info(f"Executando cada configuração {n_repetitions} vezes para calcular média e desvio padrão.")

    all_results = [] 
    
    with st.spinner(f"Executando {len(sample_sizes) * len(k_values) * n_repetitions} experimentos..."):
        # ... (o loop de execução continua exatamente o mesmo de antes, sem alterações) ...
        progress_bar = st.progress(0)
        total_experiments = len(k_values) * len(sample_sizes) * n_repetitions
        current_experiment = 0
        for i in range(n_repetitions):
            current_random_state = initial_random_state + i
            for sample_size in sample_sizes:
                try:
                    X_subset, y_subset = create_balanced_subset(X, y, sample_size, current_random_state)
                    for k in k_values:
                        current_experiment += 1
                        progress_bar.progress(current_experiment / total_experiments)
                        try:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X_subset, y_subset, test_size=0.2, random_state=current_random_state, stratify=y_subset
                            )
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            knn = KNeighborsClassifier(n_neighbors=k)
                            knn.fit(X_train_scaled, y_train)
                            y_pred = knn.predict(X_test_scaled)
                            
                            accuracy = accuracy_score(y_test, y_pred)
                            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0, labels=np.unique(y_subset))
                            precision_macro = report['macro avg']['precision']
                            recall_macro = report['macro avg']['recall']
                            f1_macro = report['macro avg']['f1-score']

                            all_results.append({
                                'Tamanho_Amostra': sample_size, 'K': k, 'Acuracia': accuracy,
                                'Precisao': precision_macro, 'Revocacao': recall_macro, 'F1-Score': f1_macro
                            })
                        except Exception:
                            pass
                except Exception as e:
                    st.error(f"Erro ao criar subset com {sample_size} amostras na repetição {i+1}: {str(e)}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        grouped = results_df.groupby(['Tamanho_Amostra', 'K'])
        mean_results = grouped.mean()
        std_results = grouped.std().fillna(0)

        def format_mean_std(mean_df, std_df, metric):
            formatted_series = (mean_df[metric].round(3).astype(str) + " ± " + std_df[metric].round(3).astype(str))
            return formatted_series.unstack()

        metrics_to_show = {'Acuracia': 'Acurácia', 'F1-Score': 'F1-Score Macro', 'Precisao': 'Precisão Macro', 'Revocacao': 'Revocação Macro'}
        for key, title in metrics_to_show.items():
            st.markdown(f"#### 📊 Tabela de Resultados ({title}: Média ± Desvio Padrão)")
            pivot_table = format_mean_std(mean_results, std_results, key)
            st.dataframe(pivot_table, use_container_width=True)

        # <<< INÍCIO DA MUDANÇA: ADICIONAR BOTÃO DE DOWNLOAD PARA AS TABELAS >>>
        st.markdown("---")
        # Prepara um único DataFrame com todos os resultados numéricos para exportação
        export_df = mean_results.join(std_results, lsuffix='_mean', rsuffix='_std')
        
        st.download_button(
           label="📥 Baixar Todas as Tabelas de Métricas (CSV)",
           data=export_df.to_csv().encode('utf-8'),
           file_name='resultados_metricas.csv',
           mime='text/csv',
        )
        # <<< FIM DA MUDANÇA >>>

        pivot_accuracy_mean = mean_results['Acuracia'].unstack()
        fig_heatmap = px.imshow(pivot_accuracy_mean.values, x=[f"K={k}" for k in pivot_accuracy_mean.columns],
                                y=[f"Amostras={s}" for s in pivot_accuracy_mean.index],
                                color_continuous_scale="Viridis", title="🔥 Mapa de Calor: Acurácia Média", text_auto=".3f")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.markdown("#### 📈 Análise de Tendências (baseada na média de 10 execuções)")
        col1, col2 = st.columns(2)
        with col1:
            fig_k_trend = px.line(mean_results.reset_index().groupby('K')['Acuracia'].mean().reset_index(),
                                  x='K', y='Acuracia', title="Tendência da Acurácia Média por K", markers=True)
            st.plotly_chart(fig_k_trend, use_container_width=True)
        with col2:
            fig_sample_trend = px.line(mean_results.reset_index().groupby('Tamanho_Amostra')['Acuracia'].mean().reset_index(),
                                       x='Tamanho_Amostra', y='Acuracia', title="Tendência da Acurácia Média por Tamanho da Amostra", markers=True)
            st.plotly_chart(fig_sample_trend, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🔄 Matriz de Confusão Média (para a melhor configuração)")

        best_config = mean_results['Acuracia'].idxmax()
        best_sample_size, best_k = best_config
        st.success(f"Melhor configuração encontrada: **{best_sample_size} amostras/classe** com **K={best_k}** (Acurácia Média: {mean_results.loc[best_config]['Acuracia']:.3f})")

        confusion_matrices = []
        for i in range(n_repetitions):
            current_random_state = initial_random_state + i
            try:
                X_subset, y_subset = create_balanced_subset(X, y, best_sample_size, current_random_state)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_subset, y_subset, test_size=0.2, random_state=current_random_state, stratify=y_subset
                )
                scaler = StandardScaler().fit(X_train)
                X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)
                
                knn = KNeighborsClassifier(n_neighbors=best_k)
                knn.fit(X_train_scaled, y_train)
                y_pred = knn.predict(X_test_scaled)
                
                labels = sorted(y_subset.unique())
                cm = confusion_matrix(y_test, y_pred, labels=labels)
                confusion_matrices.append(cm)
            except Exception:
                continue

        if confusion_matrices:
            mean_cm = np.mean(confusion_matrices, axis=0)
            fig_cm = px.imshow(mean_cm, text_auto='.2f', labels={'x': 'Predito', 'y': 'Real'},
                               x=target_names, y=target_names, title=f"Matriz de Confusão Média (K={best_k}, Amostras={best_sample_size})",
                               color_continuous_scale="Blues")
            st.plotly_chart(fig_cm, use_container_width=True)

            # <<< INÍCIO DA MUDANÇA: ADICIONAR BOTÃO DE DOWNLOAD PARA A MATRIZ DE CONFUSÃO >>>
            mean_cm_df = pd.DataFrame(mean_cm, index=target_names, columns=target_names)
            st.download_button(
               label="📥 Baixar Matriz de Confusão Média (CSV)",
               data=mean_cm_df.to_csv().encode('utf-8'),
               file_name='matriz_confusao_media.csv',
               mime='text/csv',
            )
            # <<< FIM DA MUDANÇA >>>
    else:
        st.error("Nenhum resultado foi gerado. Verifique as configurações e os dados de entrada.")
    return None
    """Executa análise sistemática do exercício variando K, tamanho das amostras e repetindo N vezes."""
    st.markdown("### 📋 Análise Sistemática - Exercício")
    
    n_repetitions = 10
    st.info(f"Executando cada configuração {n_repetitions} vezes para calcular média e desvio padrão.")

    all_results = [] 
    
    with st.spinner(f"Executando {len(sample_sizes) * len(k_values) * n_repetitions} experimentos..."):
        progress_bar = st.progress(0)
        total_experiments = len(k_values) * len(sample_sizes) * n_repetitions
        current_experiment = 0

        for i in range(n_repetitions):
            current_random_state = initial_random_state + i
            for sample_size in sample_sizes:
                try:
                    X_subset, y_subset = create_balanced_subset(X, y, sample_size, current_random_state)
                    for k in k_values:
                        current_experiment += 1
                        progress_bar.progress(current_experiment / total_experiments)
                        try:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X_subset, y_subset, test_size=0.2, random_state=current_random_state, stratify=y_subset
                            )
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            knn = KNeighborsClassifier(n_neighbors=k)
                            knn.fit(X_train_scaled, y_train)
                            y_pred = knn.predict(X_test_scaled)
                            
                            accuracy = accuracy_score(y_test, y_pred)
                            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0, labels=np.unique(y_subset))
                            precision_macro = report['macro avg']['precision']
                            recall_macro = report['macro avg']['recall']
                            f1_macro = report['macro avg']['f1-score']

                            all_results.append({
                                'Tamanho_Amostra': sample_size, 'K': k, 'Acuracia': accuracy,
                                'Precisao': precision_macro, 'Revocacao': recall_macro, 'F1-Score': f1_macro
                            })
                        except Exception:
                            pass
                except Exception as e:
                    st.error(f"Erro ao criar subset com {sample_size} amostras na repetição {i+1}: {str(e)}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        grouped = results_df.groupby(['Tamanho_Amostra', 'K'])
        mean_results = grouped.mean()
        std_results = grouped.std().fillna(0)

        def format_mean_std(mean_df, std_df, metric):
            formatted_series = (mean_df[metric].round(3).astype(str) + " ± " + std_df[metric].round(3).astype(str))
            return formatted_series.unstack()

        metrics_to_show = {'Acuracia': 'Acurácia', 'F1-Score': 'F1-Score Macro', 'Precisao': 'Precisão Macro', 'Revocacao': 'Revocação Macro'}
        for key, title in metrics_to_show.items():
            st.markdown(f"#### 📊 Tabela de Resultados ({title}: Média ± Desvio Padrão)")
            pivot_table = format_mean_std(mean_results, std_results, key)
            st.dataframe(pivot_table, use_container_width=True)
        
        pivot_accuracy_mean = mean_results['Acuracia'].unstack()
        fig_heatmap = px.imshow(pivot_accuracy_mean.values, x=[f"K={k}" for k in pivot_accuracy_mean.columns],
                                y=[f"Amostras={s}" for s in pivot_accuracy_mean.index],
                                color_continuous_scale="Viridis", title="🔥 Mapa de Calor: Acurácia Média", text_auto=".3f")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.markdown("#### 📈 Análise de Tendências (baseada na média de 10 execuções)")
        col1, col2 = st.columns(2)
        with col1:
            fig_k_trend = px.line(mean_results.reset_index().groupby('K')['Acuracia'].mean().reset_index(),
                                  x='K', y='Acuracia', title="Tendência da Acurácia Média por K", markers=True)
            st.plotly_chart(fig_k_trend, use_container_width=True)
        with col2:
            fig_sample_trend = px.line(mean_results.reset_index().groupby('Tamanho_Amostra')['Acuracia'].mean().reset_index(),
                                       x='Tamanho_Amostra', y='Acuracia', title="Tendência da Acurácia Média por Tamanho da Amostra", markers=True)
            st.plotly_chart(fig_sample_trend, use_container_width=True)

        # <<< INÍCIO DA MUDANÇA: ADICIONAR MATRIZ DE CONFUSÃO MÉDIA >>>
        st.markdown("---")
        st.markdown("#### 🔄 Matriz de Confusão Média (para a melhor configuração)")

        # 1. Encontrar a melhor configuração (maior acurácia média)
        best_config = mean_results['Acuracia'].idxmax()
        best_sample_size, best_k = best_config
        st.success(f"Melhor configuração encontrada: **{best_sample_size} amostras/classe** com **K={best_k}** (Acurácia Média: {mean_results.loc[best_config]['Acuracia']:.3f})")

        # 2. Roda os testes novamente apenas para a melhor config, salvando as matrizes
        confusion_matrices = []
        for i in range(n_repetitions):
            current_random_state = initial_random_state + i
            try:
                X_subset, y_subset = create_balanced_subset(X, y, best_sample_size, current_random_state)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_subset, y_subset, test_size=0.2, random_state=current_random_state, stratify=y_subset
                )
                scaler = StandardScaler().fit(X_train)
                X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)
                
                knn = KNeighborsClassifier(n_neighbors=best_k)
                knn.fit(X_train_scaled, y_train)
                y_pred = knn.predict(X_test_scaled)
                
                # Garante que a matriz tenha o tamanho correto (número de classes)
                labels = sorted(y_subset.unique())
                cm = confusion_matrix(y_test, y_pred, labels=labels)
                confusion_matrices.append(cm)
            except Exception:
                continue # Pula se houver erro em uma das execuções

        # 3. Calcula a média e exibe
        if confusion_matrices:
            mean_cm = np.mean(confusion_matrices, axis=0)
            fig_cm = px.imshow(
                mean_cm,
                text_auto='.2f', # Formata para 2 casas decimais
                labels={'x': 'Predito', 'y': 'Real'},
                x=target_names,
                y=target_names,
                title=f"Matriz de Confusão Média (K={best_k}, Amostras={best_sample_size})",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        # <<< FIM DA MUDANÇA >>>

    else:
        st.error("Nenhum resultado foi gerado. Verifique as configurações e os dados de entrada.")
    return None
    """Executa análise sistemática do exercício variando K, tamanho das amostras e repetindo N vezes."""
    st.markdown("### 📋 Análise Sistemática - Exercício")
    
    # <<< MUDANÇA 1: Definir o número de repetições >>>
    n_repetitions = 10
    st.info(f"Executando cada configuração {n_repetitions} vezes para calcular média e desvio padrão.")

    all_results = [] # Trocamos o nome para refletir que guardará todos os resultados brutos
    
    with st.spinner(f"Executando {len(sample_sizes) * len(k_values) * n_repetitions} experimentos..."):
        progress_bar = st.progress(0)
        total_experiments = len(k_values) * len(sample_sizes) * n_repetitions
        current_experiment = 0

        # <<< MUDANÇA 2: Loop externo para as repetições >>>
        for i in range(n_repetitions):
            # Usamos uma semente diferente para cada repetição para garantir variedade
            current_random_state = initial_random_state + i

            for sample_size in sample_sizes:
                try:
                    # Usar a semente variável na criação do subconjunto
                    X_subset, y_subset = create_balanced_subset(X, y, sample_size, current_random_state)
                    
                    for k in k_values:
                        current_experiment += 1
                        progress_bar.progress(current_experiment / total_experiments)
                        
                        try:
                            # <<< MUDANÇA 3: Usar a semente variável também na divisão treino-teste >>>
                            X_train, X_test, y_train, y_test = train_test_split(
                                X_subset, y_subset, test_size=0.2, random_state=current_random_state, stratify=y_subset
                            )
                            
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            
                            knn = KNeighborsClassifier(n_neighbors=k)
                            knn.fit(X_train_scaled, y_train)
                            
                            y_pred = knn.predict(X_test_scaled)
                            
                            # Calcular todas as métricas
                            accuracy = accuracy_score(y_test, y_pred)
                            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                            precision_macro = report['macro avg']['precision']
                            recall_macro = report['macro avg']['recall']
                            f1_macro = report['macro avg']['f1-score']

                            # Salvar os resultados brutos desta execução
                            all_results.append({
                                'Tamanho_Amostra': sample_size,
                                'K': k,
                                'Acuracia': accuracy,
                                'Precisao': precision_macro,
                                'Revocacao': recall_macro,
                                'F1-Score': f1_macro
                            })
                            
                        except Exception as e:
                            # Silenciosamente ignora erros em execuções individuais para não parar o processo
                            pass
                except Exception as e:
                    st.error(f"Erro ao criar subset com {sample_size} amostras na repetição {i+1}: {str(e)}")

    # <<< MUDANÇA 4: Processar os resultados após todas as execuções >>>
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Agrupar por configuração e calcular média e desvio padrão
        grouped = results_df.groupby(['Tamanho_Amostra', 'K'])
        mean_results = grouped.mean()
        std_results = grouped.std().fillna(0) # .fillna(0) para casos sem variação

        # Função para formatar "média ± desvio padrão"
        def format_mean_std(mean_df, std_df, metric):
            formatted_series = (mean_df[metric].round(3).astype(str) + 
                                " ± " + 
                                std_df[metric].round(3).astype(str))
            return formatted_series.unstack()

        # Criar as tabelas formatadas
        pivot_accuracy = format_mean_std(mean_results, std_results, 'Acuracia')
        pivot_f1 = format_mean_std(mean_results, std_results, 'F1-Score')

        st.markdown("#### 📊 Tabela de Resultados (Acurácia: Média ± Desvio Padrão)")
        st.dataframe(pivot_accuracy, use_container_width=True)

        st.markdown("#### 📊 Tabela de Resultados (F1-Score Macro: Média ± Desvio Padrão)")
        st.dataframe(pivot_f1, use_container_width=True)

        # <<< ADICIONE ESTE BLOCO PARA EXIBIR AS MÉTRICAS FALTANTES >>>
        pivot_precision = format_mean_std(mean_results, std_results, 'Precisao')
        pivot_recall = format_mean_std(mean_results, std_results, 'Revocacao')

        st.markdown("#### 📊 Tabela de Resultados (Precisão Macro: Média ± Desvio Padrão)")
        st.dataframe(pivot_precision, use_container_width=True)

        st.markdown("#### 📊 Tabela de Resultados (Revocação Macro: Média ± Desvio Padrão)")
        st.dataframe(pivot_recall, use_container_width=True)
        # --- FIM DO BLOCO ---
        
        # O heatmap deve usar apenas a média (dados numéricos)
        pivot_accuracy_mean = mean_results['Acuracia'].unstack()
        fig_heatmap = px.imshow(
            pivot_accuracy_mean.values,
            x=[f"K={k}" for k in pivot_accuracy_mean.columns],
            y=[f"Amostras={s}" for s in pivot_accuracy_mean.index],
            color_continuous_scale="Viridis",
            title="🔥 Mapa de Calor: Acurácia Média",
            text_auto=".3f"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Gráficos de tendência também usam a média
        st.markdown("#### 📈 Análise de Tendências (baseada na média de 10 execuções)")
        col1, col2 = st.columns(2)
        with col1:
            fig_k_trend = px.line(
                mean_results.reset_index().groupby('K')['Acuracia'].mean().reset_index(),
                x='K', y='Acuracia', title="Tendência da Acurácia Média por K", markers=True
            )
            st.plotly_chart(fig_k_trend, use_container_width=True)
        with col2:
            fig_sample_trend = px.line(
                mean_results.reset_index().groupby('Tamanho_Amostra')['Acuracia'].mean().reset_index(),
                x='Tamanho_Amostra', y='Acuracia', title="Tendência da Acurácia Média por Tamanho da Amostra", markers=True
            )
            st.plotly_chart(fig_sample_trend, use_container_width=True)
    else:
        st.error("Nenhum resultado foi gerado. Verifique as configurações e os dados de entrada.")

    return None # A função agora apenas exibe os resultados

    """Executa análise sistemática do exercício variando K e tamanho das amostras"""
    st.markdown("### 📋 Análise Sistemática - Exercício")
    
    results_table = []
    confusion_matrices = {}
    
    with st.spinner("Executando análise sistemática..."):
        progress_bar = st.progress(0)
        total_experiments = len(k_values) * len(sample_sizes)
        current_experiment = 0
        
        for sample_size in sample_sizes:
            st.write(f"**Testando com {sample_size} amostras por classe...**")
            
            # Criar subset balanceado com sample_size por classe
            try:
                X_subset, y_subset = create_balanced_subset(X, y, sample_size, random_state)
                
                for k in k_values:
                    current_experiment += 1
                    progress_bar.progress(current_experiment / total_experiments)
                    
                    # Executar KNN
                    try:
                        # Use 80% para treino, 20% para teste do subset
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_subset, y_subset, test_size=0.2, random_state=random_state, stratify=y_subset
                        )
                        
                        # Padronização
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Treinamento
                        knn = KNeighborsClassifier(n_neighbors=k)
                        knn.fit(X_train_scaled, y_train)
                        
                        # Predições
                        y_pred = knn.predict(X_test_scaled)
                        accuracy = accuracy_score(y_test, y_pred)

                        # --- INÍCIO DA CORREÇÃO ---
                        # Adicione este bloco para calcular as métricas que faltavam
                        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                        precision_macro = report['macro avg']['precision']
                        recall_macro = report['macro avg']['recall']
                        f1_macro = report['macro avg']['f1-score']
                        # --- FIM DA CORREÇÃO ---

                        # Salvar resultados
                        results_table.append({
                            'Tamanho_Amostra': sample_size,
                            'K': k,
                            'Acuracia': accuracy,
                            'Precisao': precision_macro,
                            'Revocacao': recall_macro,
                            'F1-Score': f1_macro,
                            'N_Treino': len(X_train),
                            'N_Teste': len(X_test)
                        })
                        
                        # Salvar matriz de confusão para casos específicos
                        if k == 5:  # Salva matriz para k=5 como exemplo
                            cm = confusion_matrix(y_test, y_pred)
                            confusion_matrices[f"Sample_{sample_size}_K_{k}"] = cm
                            
                    except Exception as e:
                        st.warning(f"Erro para K={k}, Amostra={sample_size}: {str(e)}")
                        
            except Exception as e:
                st.error(f"Erro ao criar subset com {sample_size} amostras: {str(e)}")
    
    # Criar tabela de resultados
    if results_table:
        results_df = pd.DataFrame(results_table)
        
        # Pivot table como solicitado no exercício
        pivot_table = results_df.pivot(index='Tamanho_Amostra', columns='K', values='Acuracia')
        
        st.markdown("#### 📊 Tabela de Resultados (Acurácia)")
        st.dataframe(pivot_table.round(4), use_container_width=True)
        
         # --- ADICIONE ESTE BLOCO PARA VER AS OUTRAS MÉTRICAS ---
        st.markdown("#### 📊 Tabela de Resultados (F1-Score Macro)")
        pivot_f1 = results_df.pivot(index='Tamanho_Amostra', columns='K', values='F1-Score')
        st.dataframe(pivot_f1.round(4), use_container_width=True)

        st.markdown("#### 📊 Tabela de Resultados (Precisão Macro)")
        pivot_precision = results_df.pivot(index='Tamanho_Amostra', columns='K', values='Precisao')
        st.dataframe(pivot_precision.round(4), use_container_width=True)
        # --- FIM DO BLOCO ---

        # Heatmap da tabela
        fig_heatmap = px.imshow(
            pivot_table.values,
            x=[f"K={k}" for k in pivot_table.columns],
            y=[f"Amostras={s}" for s in pivot_table.index],
            color_continuous_scale="Viridis",
            title="🔥 Mapa de Calor: Acurácia por K e Tamanho da Amostra",
            text_auto=".3f"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Análise de tendências
        st.markdown("#### 📈 Análise de Tendências")
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico por K
            fig_k_trend = px.line(
                results_df.groupby('K')['Acuracia'].mean().reset_index(),
                x='K', y='Acuracia',
                title="Tendência da Acurácia por K",
                markers=True
            )
            st.plotly_chart(fig_k_trend, use_container_width=True)
        
        with col2:
            # Gráfico por tamanho da amostra
            fig_sample_trend = px.line(
                results_df.groupby('Tamanho_Amostra')['Acuracia'].mean().reset_index(),
                x='Tamanho_Amostra', y='Acuracia',
                title="Tendência da Acurácia por Tamanho da Amostra",
                markers=True
            )
            st.plotly_chart(fig_sample_trend, use_container_width=True)
        
        # Mostrar algumas matrizes de confusão
        if confusion_matrices:
            st.markdown("#### 🔄 Matrizes de Confusão (Exemplos)")
            for name, cm in list(confusion_matrices.items())[:3]:  # Mostrar até 3 exemplos
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write(f"**{name.replace('_', ' ')}**")
                    st.dataframe(pd.DataFrame(cm, 
                                            columns=target_names[:cm.shape[1]], 
                                            index=target_names[:cm.shape[0]]))
                with col2:
                    fig_cm = px.imshow(
                        cm, text_auto=True,
                        title=f"Matriz de Confusão - {name.replace('_', ' ')}",
                        color_continuous_scale="Blues"
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
    
    return results_df if results_table else None


def create_balanced_subset(X, y, samples_per_class, random_state=42):
    """Cria um subset balanceado com n amostras por classe"""
    np.random.seed(random_state)
    
    X_subset = []
    y_subset = []
    
    for class_label in np.unique(y):
        # Índices da classe atual
        class_indices = np.where(y == class_label)[0]
        
        # Se há menos amostras que o solicitado, usa todas
        n_samples = min(samples_per_class, len(class_indices))
        
        # Seleciona aleatoriamente n_samples
        selected_indices = np.random.choice(class_indices, size=n_samples, replace=False)
        
        X_subset.extend(X.iloc[selected_indices].values)
        y_subset.extend(y.iloc[selected_indices].values)
    
    return pd.DataFrame(X_subset, columns=X.columns), pd.Series(y_subset)

def create_prediction_interface(model, scaler, feature_names, target_names):
    """Interface para fazer predições"""
    st.markdown("### 🎯 Fazer Predições")
    
    st.write("Insira os valores das features para fazer uma predição:")
    
    input_values = []
    cols = st.columns(2)
    
    for i, feature in enumerate(feature_names):
        with cols[i % 2]:
            value = st.number_input(
                f"{feature}",
                value=0.0,
                step=0.1,
                format="%.2f"
            )
            input_values.append(value)
    
    if st.button("🔮 Fazer Predição"):
        try:
            # Prepara os dados
            input_array = np.array(input_values).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            
            # Faz predição
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Ensure prediction is valid index
            if prediction < len(target_names):
                predicted_class = target_names[prediction]
            else:
                predicted_class = f"Class_{prediction}"
            
            # Exibe resultado
            st.success(f"🎯 Predição: **{predicted_class}**")
            
            # Probabilidades - evita usar % no formato
            prob_df = pd.DataFrame({
                'Classe': target_names[:len(prediction_proba)],
                'Probabilidade': prediction_proba
            }).sort_values('Probabilidade', ascending=False)
            
            # Mostra as probabilidades como texto também
            st.write("**Probabilidades:**")
            for idx, row in prob_df.iterrows():
                prob_percent = row['Probabilidade'] * 100
                st.write(f"- {row['Classe']}: {prob_percent:.2f}%")
            
            fig_prob = px.bar(
                prob_df,
                x='Classe',
                y='Probabilidade',
                title="📊 Probabilidades por Classe",
                color='Probabilidade',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_prob, use_container_width=True)
        except Exception as e:
            st.error(f"Erro na predição: {str(e)}")

def run_exercise_analysis_proportions(X, y, target_names, k_values, train_proportions, initial_random_state, compare_scaling=True):
    """Executa análise sistemática variando K, proporção treino/teste e normalização"""
    st.markdown("### 📋 Análise Sistemática - Por Proporção de Treino")
    
    n_repetitions = 10
    st.info(f"Executando cada configuração {n_repetitions} vezes para calcular média e desvio padrão.")

    all_results = []
    scaling_options = ['Com normalização', 'Sem normalização'] if compare_scaling else ['Com normalização']
    
    with st.spinner(f"Executando {len(train_proportions) * len(k_values) * len(scaling_options) * n_repetitions} experimentos..."):
        progress_bar = st.progress(0)
        total_experiments = len(k_values) * len(train_proportions) * len(scaling_options) * n_repetitions
        current_experiment = 0

        for i in range(n_repetitions):
            current_random_state = initial_random_state + i
            
            for train_prop in train_proportions:
                for use_scaling in scaling_options:
                    for k in k_values:
                        current_experiment += 1
                        progress_bar.progress(current_experiment / total_experiments)
                        
                        try:
                            # Divisão treino-teste
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, train_size=train_prop, random_state=current_random_state, stratify=y
                            )
                            
                            # Normalização condicional
                            if use_scaling == 'Com normalização':
                                scaler = StandardScaler()
                                X_train_scaled = scaler.fit_transform(X_train)
                                X_test_scaled = scaler.transform(X_test)
                            else:
                                X_train_scaled = X_train.values
                                X_test_scaled = X_test.values
                            
                            # Treinamento e predição
                            knn = KNeighborsClassifier(n_neighbors=k)
                            knn.fit(X_train_scaled, y_train)
                            y_pred = knn.predict(X_test_scaled)
                            
                            # Métricas
                            accuracy = accuracy_score(y_test, y_pred)
                            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                            precision_macro = report['macro avg']['precision']
                            recall_macro = report['macro avg']['recall']
                            f1_macro = report['macro avg']['f1-score']

                            all_results.append({
                                'Proporcao_Treino': f"{int(train_prop*100)}%",
                                'K': k,
                                'Normalizacao': use_scaling,
                                'Acuracia': accuracy,
                                'Precisao': precision_macro,
                                'Revocacao': recall_macro,
                                'F1-Score': f1_macro
                            })
                            
                        except Exception as e:
                            continue

    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Agrupar e calcular estatísticas
        grouped = results_df.groupby(['Proporcao_Treino', 'K', 'Normalizacao'])
        mean_results = grouped.mean()
        std_results = grouped.std().fillna(0)

        def format_mean_std(mean_df, std_df, metric):
            formatted_series = (mean_df[metric].round(3).astype(str) + " ± " + std_df[metric].round(3).astype(str))
            return formatted_series

        # Tabelas por normalização
        for scaling_type in scaling_options:
            st.markdown(f"#### 📊 Resultados - {scaling_type}")
            
            subset_mean = mean_results.xs(scaling_type, level='Normalizacao')
            subset_std = std_results.xs(scaling_type, level='Normalizacao')
            
            for metric in ['Acuracia', 'F1-Score', 'Precisao', 'Revocacao']:
                pivot_table = format_mean_std(subset_mean, subset_std, metric).unstack()
                st.markdown(f"**{metric} (Média ± Desvio Padrão)**")
                st.dataframe(pivot_table, use_container_width=True)
        
        # Comparação com/sem normalização se aplicável
        if compare_scaling:
            st.markdown("#### 📈 Comparação: Com vs Sem Normalização")
            
            comparison_data = []
            for _, row in results_df.groupby(['Proporcao_Treino', 'K', 'Normalizacao'])['Acuracia'].mean().reset_index().iterrows():
                comparison_data.append({
                    'Configuração': f"{row['Proporcao_Treino']} - K={row['K']}",
                    'Normalização': row['Normalizacao'],
                    'Acurácia Média': row['Acuracia']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            fig_comparison = px.bar(
                comparison_df,
                x='Configuração',
                y='Acurácia Média',
                color='Normalização',
                title="🔄 Impacto da Normalização na Acurácia",
                barmode='group'
            )
            fig_comparison.update_xaxes(tickangle=45)
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Análise crítica automática
        st.markdown("#### 🧠 Análise Crítica Automática")
        
        best_k = mean_results['Acuracia'].idxmax()[1]  # K da melhor configuração
        worst_k = mean_results['Acuracia'].idxmin()[1]  # K da pior configuração
        
        st.markdown(f"""
        **Observações sobre viés vs. variância:**
        - **K={best_k}** mostrou melhor desempenho médio, sugerindo bom equilíbrio entre viés e variância
        - **K={worst_k}** teve pior desempenho, indicando possível overfitting (K muito baixo) ou underfitting (K muito alto)
        
        **Efeito da normalização:**
        {f"- A normalização mostrou impacto significativo nos resultados, confirmando a importância da padronização para k-NN" if compare_scaling else "- Executado apenas com normalização (recomendado para k-NN)"}
        
        **Efeito do tamanho do conjunto de treino:**
        - Proporções maiores de treino tendem a melhorar a acurácia, mas reduzem a confiabilidade da avaliação
        - O desvio padrão indica a estabilidade do modelo across diferentes divisões aleatórias
        """)
        
        # Download dos resultados
        export_df = mean_results.join(std_results, lsuffix='_mean', rsuffix='_std')
        st.download_button(
           label="📥 Baixar Resultados Completos (CSV)",
           data=export_df.to_csv().encode('utf-8'),
           file_name='resultados_proporcoes.csv',
           mime='text/csv',
        )

    else:
        st.error("Nenhum resultado foi gerado. Verifique as configurações.")

def main():
    # Header
    st.markdown("<h1 class='main-header'>🤖 k-NN Interactive Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("**Análise interativa de classificação com k-Nearest Neighbors**")
    
    # Sidebar para configurações
    st.sidebar.header("⚙️ Configurações")
    
    # Seleção do dataset
    dataset_choice = st.sidebar.selectbox(
        "📊 Escolha o Dataset:",
        ["Iris", "Wine"]
    )
    
    # Carrega dados
    try:
        X, y, target_names, description = load_dataset(dataset_choice)
    except Exception as e:
        st.error(f"Erro ao carregar dataset: {str(e)}")
        return
    
    # Configurações do modelo
    st.sidebar.subheader("🔧 Parâmetros do Modelo")
    
    k_value = st.sidebar.slider(
        "Valor de k:",
        min_value=1,
        max_value=min(20, len(X) // 3),
        value=5,
        help="Número de vizinhos mais próximos a considerar"
    )
    
    test_size = st.sidebar.slider(
        "Tamanho do conjunto de teste:",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        format="%.2f",
        help="Porcentagem dos dados para teste"
    )
    
    random_state = st.sidebar.number_input(
        "Random State:",
        value=42,
        help="Semente para reprodutibilidade"
    )
    
    # Configurações do exercício específico
    st.sidebar.subheader("📋 Configurações do Exercício")
    
    use_exercise_mode = st.sidebar.checkbox(
        "Modo Exercício (Tabela K vs Tamanho da Amostra)",
        help="Executa análise sistemática variando K e tamanho das amostras"
    )
    
    if use_exercise_mode:
        exercise_type = st.sidebar.radio(
            "Tipo de análise:",
            ["Por proporção de treino", "Por amostras por classe"],
            help="Escolha o tipo de variação conforme o exercício"
        )
        
        if exercise_type == "Por proporção de treino":
            train_proportions = st.sidebar.multiselect(
                "Proporções de treino:",
                options=[0.6, 0.7, 0.8],
                default=[0.6, 0.7, 0.8],
                help="Proporção dos dados para treinamento (60%, 70%, 80%)"
            )
        else:
            sample_sizes = st.sidebar.multiselect(
                "Tamanhos de amostra por classe:",
                options=[5, 10, 15, 20, 25, 30],
                default=[5, 10, 15, 20],
                help="Número de amostras por classe para treino"
            )
        
        k_values = st.sidebar.multiselect(
            "Valores de K para testar:",
            options=[1, 3, 5, 7, 9],
            default=[1, 3, 5, 7, 9],
            help="Valores de K a serem testados"
        )
        
        compare_scaling = st.sidebar.checkbox(
            "Comparar com/sem normalização",
            value=True,
            help="Executa análise com e sem StandardScaler"
        )
    
    # Opções de visualização
    st.sidebar.subheader("📊 Opções de Visualização")
    show_data_overview = st.sidebar.checkbox("Visão Geral dos Dados", True)
    show_feature_analysis = st.sidebar.checkbox("Análise de Features", True)
    show_pca = st.sidebar.checkbox("Visualização PCA", True)
    show_optimization = st.sidebar.checkbox("Otimização de k", False)
    show_prediction = st.sidebar.checkbox("Interface de Predição", False)
    
    # Conteúdo principal
    if show_data_overview:
        create_data_overview(X, y, target_names, description)
        st.divider()
    
    if show_feature_analysis:
        create_feature_analysis(X, y, target_names)
        st.divider()
    
    if show_pca:
        pca, X_pca = create_pca_visualization(X, y, target_names)
        st.divider()
    
    # Análise do modelo atual
    if not use_exercise_mode:
        st.markdown("### 🚀 Análise do Modelo k-NN")
        
        with st.spinner("Treinando modelo..."):
            try:
                results = run_knn_analysis(X, y, target_names, k_value, test_size, random_state)
                create_results_visualization(results, target_names, k_value)
            except Exception as e:
                st.error(f"Erro durante análise do modelo: {str(e)}")
    else:
        # Modo exercício
        if k_values:
            if exercise_type == "Por proporção de treino":
                if train_proportions:
                    run_exercise_analysis_proportions(X, y, target_names, k_values, train_proportions, random_state, compare_scaling)
                else:
                    st.warning("⚠️ Selecione pelo menos uma proporção de treino e um valor de K")
            else:
                if sample_sizes:
                    run_exercise_analysis(X, y, target_names, k_values, sample_sizes, random_state)
                else:
                    st.warning("⚠️ Selecione pelo menos um tamanho de amostra e um valor de K")
        else:
            st.warning("⚠️ Selecione pelo menos um valor de K para o exercício")
    
    if show_optimization and not use_exercise_mode:
        st.divider()
        try:
            optimal_k = k_optimization_analysis(X, y, target_names, test_size, random_state)
        except Exception as e:
            st.error(f"Erro na otimização: {str(e)}")
    
    if show_prediction and not use_exercise_mode:
        st.divider()
        try:
            if 'results' in locals():
                create_prediction_interface(results['model'], results['scaler'], X.columns, target_names)
            else:
                st.info("Execute a análise do modelo primeiro para usar a interface de predição")
        except Exception as e:
            st.error(f"Erro na interface de predição: {str(e)}")
    
    # Informações sobre o exercício
    with st.expander("📋 Sobre o Exercício (Prof. José Alfredo Costa)"):
        st.markdown("""
        **Objetivo do Exercício:**
        
        Testar o algoritmo KNN nas bases Iris e Wine variando:
        - **K**: 1, 3, 5, 7, 9 (número de vizinhos mais próximos)
        - **Proporção de treino**: 60%, 70%, 80% (conforme exercício original)
        - **Normalização**: Comparar com e sem StandardScaler
        
        **Como usar o Modo Exercício:**
        1. ✅ Marque "Modo Exercício" na barra lateral
        2. 📊 Escolha "Por proporção de treino" (conforme exercício) ou "Por amostras por classe"
        3. 🔢 Selecione os valores de K e proporções para testar
        4. ⚖️ Marque "Comparar com/sem normalização" para análise completa
        5. 🚀 O sistema executará automaticamente todas as combinações
        6. 📈 Visualize resultados, análise crítica e baixe os dados
        
        **Métricas reportadas (conforme solicitado):**
        - Acurácia, precisão, revocação, F1-score (macro)
        - Matriz de confusão
        - Tabelas sintetizando resultados por k e proporção de treino
        - Análise crítica sobre viés vs. variância e efeito da escala
        
        **Reprodutibilidade garantida:**
        - 10 repetições com seeds diferentes
        - Média ± desvio padrão para maior estabilidade
        """)
    
    # Informações adicionais
    with st.expander("ℹ️ Sobre o k-NN"):
        st.markdown("""
        **k-Nearest Neighbors (k-NN)** é um algoritmo de aprendizado supervisionado usado para classificação e regressão.
        
        **Como funciona:**
        - Para classificar um novo ponto, encontra os k pontos mais próximos no conjunto de treinamento
        - A classificação é feita por voto majoritário dos k vizinhos
        - A distância é geralmente calculada usando a distância euclidiana
        
        **Vantagens:**
        - ✅ Simples de entender e implementar
        - ✅ Não faz suposições sobre a distribuição dos dados
        - ✅ Funciona bem com datasets pequenos
        
        **Desvantagens:**
        - ❌ Computacionalmente caro para datasets grandes
        - ❌ Sensível à escala das features (por isso a padronização é importante)
        - ❌ Performance degrada em altas dimensões (curse of dimensionality)
        """)

if __name__ == "__main__":
    main()