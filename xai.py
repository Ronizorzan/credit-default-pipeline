import io
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
import numpy as np

from textwrap import dedent
from joblib import load
from agno.agent import Agent
from agno.media import Image as AgnoImage
from agno.models.google import Gemini
from dotenv import load_dotenv

load_dotenv()

# Columns mapper
mapper_columns = {
    "student_target_enc": "Estudante",
    "balance_bin": "Saldo Devedor Atual",
    "income": "Renda Anual",
    "balance_over_mean_income": "Saldo Devedor vs Renda Média",
    "balance_warning_zone": "Saldo Devedor 1000 - 2000",
    "balance_income_ratio": "Razão Saldo Devedor vs Renda"
}

# -------------------------
# Layout configuration
# -------------------------
st.set_page_config("Inteligência de Risco & Transparência", page_icon="🛡️", layout="wide")
st.sidebar.title("🛡️ Decisões Transparentes")
st.sidebar.image("https://th.bing.com/th/id/OIG4.lgGfp80wbrub3nvTrEpw?pid=ImgGn", width=200)
st.sidebar.markdown("<hr style='border: 1px solid; color: #03F277'>", unsafe_allow_html=True)
st.sidebar.title("⚙️ Painel de Controle")

with st.sidebar.expander("Selecione a Análise"):
    graph_type = st.radio("Foco da Interpretação ->",
                                options=["Visão Global de Risco", "Auditoria Individual", "Análise Multidimensional", "Interação de Fatores"], 
                                help="Visão Global de Risco -> Entenda o comportamento geral da carteira.\
                                \nAuditoria Individual -> Avalie o perfil de risco de um cliente específico.\
                                \nAnálise Multidimensional -> Distribuição completa dos fatores de risco.\
                                \nInteração de Fatores -> Descubra padrões complexos entre variáveis.")
    
    if graph_type=="Auditoria Individual":
        idx = st.number_input("Selecione o ID do Cliente (Dossiê)", 1, 20000, 10)
    if graph_type=="Interação de Fatores":
        dependence_column = st.selectbox("Selecione o Fator Principal para gerar Análise Cruzada", options=mapper_columns.values())

generate = st.sidebar.button("🚀 Extrair Insights Estratégicos", type="primary", use_container_width=True)

# -------------------------
# 1. Load test data
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/test_processed.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y

# -------------------------
# 2. Load pre-trained model
# -------------------------
@st.cache_resource
def load_xgb_model():
    xgb_model = load("models/xgb_model.joblib")
    return xgb_model

# ----------------------------------
# 2.1 Load Google LLM with Gemini
# ----------------------------------
@st.cache_resource
def load_llm():
    model_id = "gemini-2.5-flash"
    project_agent = Agent(model=Gemini(id=model_id, project_id="projetos-python-490617"),
                          name="Risk Strategy Agent",
                          role="Chief Risk Officer (CRO) Especialista em IA e Mitigação de Risco de Crédito",
                          markdown=True,
                          instructions=dedent(
                            """                            
                            Atue como um Especialista Sênior em Risco de Crédito explicando o modelo preditivo de inadimplência (default em cartão de crédito) para uma diretoria não-técnica.
                            Gere um Relatório Executivo baseado no gráfico de interpretabilidade SHAP fornecido.
                            
                            **Diretrizes de Tom e Formatação:**
                            - Use uma linguagem sofisticada, orientada a negócios, focada em impacto, ações e prevenção de perdas.
                            - Estruture a resposta com cabeçalhos curtos para exibição em espaço limitado, bullet points e destaques em negrito para facilitar a leitura dinâmica.
                            - Não adicione datas ou qualquer outro tipo de variável entre [].
                            - NUNCA use jargões técnicos excessivos (como 'valores SHAP' ou 'log odds') sem explicar rapidamente seu impacto prático.
                            - Ressalte a ética: Deixe claro em uma nota de rodapé que o modelo analisa exclusivamente variáveis transacionais e financeiras comportamentais,
                              garantindo total imparcialidade e evitando vieses discriminatórios.
                            """
                          )
                          )
    return project_agent

# ----------------------------
# 3. Load process and cache resources
# ----------------------------
xgb_model = load_xgb_model()
agent = load_llm()
X, y = load_data()
X.rename(columns=mapper_columns, inplace=True)
inverse_mapper = {1: "Sim", 0: "Não"}


# -----------------------------
# 4. Explainability with SHAP
# -----------------------------
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X)

if "mean_importance_analysis" not in st.session_state:
    st.session_state["mean_importance_analysis"] = None

if "summary_plot" not in st.session_state:
    st.session_state["summary_plot"] = None



# --------------------------------------
# 5. Mean-Importance-plot - Interative graph
# --------------------------------------
if graph_type=="Visão Global de Risco" and generate:    
    st.markdown("<h1 style='text-align: center'>Principais Impulsionadores de Inadimplência</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid; color: #03F277'>", unsafe_allow_html=True)
    graph_col, explain_col = st.columns([0.45, 0.45])    
    
    try:
        mean_shap = pd.DataFrame(
            {"feature": X.columns,
            "importance": abs(shap_values).mean(axis=0)}
        ).sort_values("importance", ascending=False).head(4)

        fig = px.bar(mean_shap.round(2), x="importance", y="feature",
                    labels={"importance": "Importância Média", "feature": "Característica"},
                    title="Quais as características mais importantes", orientation="h",
                    color="importance", color_continuous_scale="Greens"
                    )
        
        with graph_col:
            st.header("Principais Indicadores de Inadimplência")
            st.plotly_chart(fig, width="stretch")

        with explain_col:
            st.subheader("💡 Relatório executivo")
            if st.session_state["mean_importance_analysis"] is not None:
                st.markdown(st.session_state["mean_importance_analysis"])
            
            else:
                with st.spinner("🧠 Gerando Interpretação do Gráfico com IA"):
                    response = agent.run(input=f"""Analise os dados globais  de risco e resuma os top 3 fatores críticos que impulsionam o 'default' na carteira.
                                    Mostre quais variáveis têm maior impacto nas decisões do Modelo e quais conclusões e recomendações
                                    pode se extrair desses dados.
                                    "Dados do Gráfico Plotly": {mean_shap},
                                    "Colunas": {list(X.columns)},
                                    "Mapa de Cores": "Greens"
                                    """)
            
                    st.session_state["mean_importance_analysis"] = response.content
                st.markdown(st.session_state["mean_importance_analysis"])

    except Exception as error:
        st.error(f"""❌ Ocorreu um erro ao processar sua requisição. 
                    Pedimos desculpas pelo inconveniente.

                    🔄 Por favor, tente novamente em alguns instantes.  
                    📝 Se o problema persistir, você pode registrar um comentário através do formulário:  
                    http://100.54.239.46:5678/form/76aa90fc-ad5f-45fd-af23-4a09c8317016  

                    📌 Detalhes técnicos do erro: {error}
                    """
                    )
        
# --------------------------------------
# 6. Force-plot - Interative graph
# --------------------------------------
elif graph_type=="Auditoria Individual" and generate:
    st.markdown("<h1 style='text-align: center'>📑 Dossiê de Decisão: Perfil de Risco Individual</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid; color: #03F277'>", unsafe_allow_html=True)
    graph_col, explain_col = st.columns([0.45, 0.55])    
    try:        
        selected_instance = X.iloc[idx]        
        predicted = xgb_model.predict(np.reshape(selected_instance.values, (1, -1)))
        y_selected = y[idx]

        shap_instance = shap_values[idx]
        instance_df = pd.DataFrame({
            "feature": X.columns,
            "value": selected_instance.values,
            "shap_value": shap_instance
        }).sort_values("shap_value", ascending=True)

        fig2 = go.Figure(go.Bar(
            x=instance_df["shap_value"],
            y=instance_df["feature"],
            orientation="h",
            marker=dict(color=instance_df["shap_value"], colorscale="Greens")
        ))

        fig2.update_layout(title=f"Decisão do Algoritmo - Previsto: {inverse_mapper[predicted[0]]} - Fato Real: {inverse_mapper[y_selected]}",
                        xaxis_title="Influência na Decisão",
                        yaxis_title="Variáveis")
        
        with graph_col:  
            st.header(f"📊 Interpretação para o cliente ID: {idx}")
            st.plotly_chart(fig2, width="stretch")
        
        with explain_col:
            #st.subheader("💡 Relatório executivo")
            with st.spinner("🔎 Gerando análise explicativa..."):
                response = agent.run(input=f"""
                    Explique de forma clara e impactante quais fatores tiveram maior influência na decisão do modelo
                    para o cliente analisado. Relacione os resultados previstos com os resultados reais,
                    destacando os principais drivers que orientam o risco individual.
                    Dados do gráfico: {instance_df.to_dict(orient="records")}
                    Valor previsto: {inverse_mapper[predicted[0]]}
                    Valor real: {inverse_mapper[y_selected]}
                """)
                                
                st.markdown(response.content)    
    
    except Exception as error:
        st.error(f"""❌ Ocorreu um erro ao gerar a análise. 
                    🔄 Por favor, tente novamente em alguns instantes.  
                    📝 Se o problema persistir, registre um comentário através do formulário:   
                    http://100.54.239.46:5678/form/76aa90fc-ad5f-45fd-af23-4a09c8317016   
                    📌 Detalhes técnicos: {error}
                    """)
        

# --------------------------------------
# 7. Summary Plot - Análise Multidimensional
# --------------------------------------
elif graph_type=="Análise Multidimensional" and generate:
    st.markdown("<h1 style='text-align: center'>🌐 Principais Fatores que Influenciam o Modelo", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid; color: #03F277'>", unsafe_allow_html=True)
    try:
        with st.spinner("🔎 Gerando análise dos fatores..."):
            graph_col, explain_col = st.columns([0.45, 0.55], gap="medium")
            
            fig3, ax = plt.subplots()
            shap.summary_plot(shap_values[:500], X[:500], show=False)
            
            buffer = io.BytesIO()
            fig3.savefig(buffer, format="png")
            buffer.seek(0)        
        
            agno_fig = AgnoImage(content=buffer.read())

            with graph_col:
                st.header("📈 Gráfico de Importância das Variáveis")
                st.pyplot(fig3, width="stretch")
            
            with explain_col:
                #st.subheader("💡 Relatório executivo")
                if st.session_state["summary_plot"] is None:
                    response = agent.run(input="""Explique o gráfico de forma clara e objetiva para executivos.
                        Destaque os fatores mais relevantes que influenciam o modelo e que devem ser acompanhados
                        para melhorar resultados. Traduza os achados em insights estratégicos que apoiem decisões de negócio,
                        evitando jargões técnicos.""",
                                        images=[agno_fig])
                    
                    st.session_state["summary_plot"] = response.content            
                    st.markdown(response.content)
                
                else:
                    st.markdown(st.session_state["summary_plot"])
            
    
    except Exception as error:
        st.error(f"""❌ Ocorreu um erro ao gerar a análise. 
                    🔄 Por favor, tente novamente em alguns instantes.  
                    📝 Se o problema persistir, registre um comentário através do formulário:   
                    http://100.54.239.46:5678/form/76aa90fc-ad5f-45fd-af23-4a09c8317016   
                    📌 Detalhes técnicos: {error}
                    """)
        
        
# --------------------------------------
# 8. Dependence Plot - Interação de Fatores
# --------------------------------------
elif graph_type=="Interação de Fatores" and generate:        
    st.markdown("<h1 style='text-align: center; color: 2C3E50'>🔗 Interação entre Variáveis e Impacto no Modelo</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid; color: #03F277'>", unsafe_allow_html=True)
    try:
        with st.spinner("🔎 Gerando análise das interações..."):
            graph_col, explain_col = st.columns([0.45, 0.55], gap="large")
            
            fig4, ax = plt.subplots()            
            shap.dependence_plot(dependence_column, shap_values, X, ax=ax, feature_names=X.columns)
            
            buffer = io.BytesIO()
            fig4.savefig(buffer, format="png")
            buffer.seek(0)        
        
            agno_dependence_fig = AgnoImage(content=buffer.read())

            with graph_col:
                st.header("📈 Gráfico de Interação entre Variáveis")
                st.pyplot(fig4, width="stretch")
            
            with explain_col:
                #st.subheader("💡 Relatório executivo")
                response = agent.run(input=dedent("""Explique o gráfico de forma clara e objetiva para executivos.
                                    Mostre como a interação entre fatores influencia os resultados do modelo.
                                    O objetivo é gerar insights estratégicos que apoiem decisões de negócio,
                                    destacando quais combinações de variáveis têm maior impacto.
                                            """),
                                    images=[agno_dependence_fig])
                                    
                st.markdown(response.content)
                                
            
    
    except Exception as error:
        st.error(f"""❌ Ocorreu um erro ao gerar a análise. 
                    🔄 Por favor, tente novamente em alguns instantes.  
                    📝 Se o problema persistir, registre um comentário através do formulário:   
                    http://100.54.239.46:5678/form/76aa90fc-ad5f-45fd-af23-4a09c8317016
                    --------
                    📌 Detalhes técnicos: {error}
                    """)
    
        







