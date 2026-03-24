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

# -------------------------
# Layout configuration
# -------------------------
st.set_page_config("Interpretabilidade de Modelo", page_icon="robot", layout="wide")
st.sidebar.title("⚙️ Configurações")
graph_type = st.sidebar.radio("Escolha o tipo de Explicação", options=["Importância Média", "Força de Explicação", "Summary Plot"])
if graph_type=="Força de Explicação":
    idx = st.sidebar.number_input("Selectione o ID do cliente", 1, 20000, 10)
generate = st.sidebar.button("Gerar interpretação personalizada", type="primary")

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
#@st.cache_resource
def load_llm():
    model_id = "gemini-2.5-flash"
    project_agent = Agent(model=Gemini(id=model_id, project_id="projetos-python-490617"),
                          name="Interpretability Agent",
                          role="Especialista em Explicabilidade de Modelos baseados em árvores",
                          markdown=True,
                          instructions=dedent(
                            """                            
                            Gere um relatório executivo desse gráfico de Interpretabilidade do XGBoost com shap_values.
                            Mantenha a linguagem direta, resumida, impactante e acessível, com foco em insights estratégicos para o negócio.                            
                            O modelo prevê a probabilidade de 'default' em cartão de crédito.
                            Ressalte que as análises são puramente técnicas e não tem o intuito, de forma alguma,
                            de discriminar quaisquer tipos de classes sociais, sejam elas minoritárias ou não.
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
mapper_columns = {
    "student_target_enc": "Estudante Universitário",
    "balance_bin": "Saldo Devedor Atual",
    "income": "Renda Anual",
    "balance_over_mean_income": "Saldo Devedor vs Renda Média",
    "balance_warning_zone": "Saldo Devedor 1000 - 2000",
    "balance_income_ratio": "Razão Saldo Devedor vs Renda"
}
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
if graph_type=="Importância Média" and generate:    
    graph_col, explain_col = st.columns([0.55, 0.45])    
    
    try:
        mean_shap = pd.DataFrame(
            {"feature": X.columns,
            "importance": abs(shap_values).mean(axis=0)}
        ).sort_values("importance", ascending=False)

        fig = px.bar(mean_shap.round(2), x="importance", y="feature",
                    labels={"importance": "Importância Média", "feature": "Característica"},
                    title="Quais as características mais importantes", orientation="h",
                    color="importance", color_continuous_scale="Greens"
                    )
        
        with graph_col:
            st.subheader("Features Importantes - Segundo o Modelo")
            st.plotly_chart(fig, width="stretch")

        with explain_col:
            if st.session_state["mean_importance_analysis"] is not None:
                st.markdown(st.session_state["mean_importance_analysis"])
            
            else:
                with st.spinner("Gerando Gráfico de Importância Média"):
                    response = agent.run(input=f"""Explique em linguagem simples o seguinte gráfico SHAP de Importância Média.
                                    Mostre quais variáveis têm maior impacto nas decisões do Modelo e quais conclusões e recomendações
                                    pode se extrair desses dados. Mantenha a análise simples e direta, sem torná-la extensa.
                                    Ideal para ser exibido em um relatório executivo para stakeholders.                                
                                    Baseie-se estritamente nos dados abaixo para embasar suas análises.
                                    "Dados do Gráfico Plotly": {mean_shap},
                                    "Colunas": {list(X.columns)},
                                    "Mapa de Cores": "Greens"
                                    """)
            
                    st.session_state["mean_importance_analysis"] = response.content
                st.markdown(st.session_state["mean_importance_analysis"])

    except Exception as error:
        st.error(f"""Erro ao processar requisição. Tente novamente...\nErro: {error}.
                 Problemas ao utilizar o projeto? Deixe um comentário através do 
                 formulário: http://100.54.239.46:5678/form/76aa90fc-ad5f-45fd-af23-4a09c8317016 """)    

# --------------------------------------
# 6. Force-plot - Interative graph
# --------------------------------------
elif graph_type=="Força de Explicação" and generate:
    graph_col, explain_col = st.columns([0.55, 0.45])    
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

        fig2.update_layout(title=f"Previsto: {inverse_mapper[predicted[0]]} - Real: {inverse_mapper[y_selected]}",
                        xaxis_title="Impacto Médio no Modelo",
                        yaxis_title="Feature")
        
        with graph_col:  
            st.subheader(f"Interpretação de resultado para o cliente ID: {idx}")
            st.plotly_chart(fig2, width="stretch")
        
        with explain_col:
            with st.spinner("Gerando Gráfico e Interpretação"):
                response = agent.run(input=f"""Explique em linguagem simples o seguinte gráfico SHAP de Força de Explicação.
                            Mostre, de forma clara e impactante, quais variáveis têm maior impacto nas decisões do Modelo
                            sobre o cliente em questão e como isso se alinha com com o Previsto X Real.
                            Mantenha a análise simples e direta, sem torná-la extensa.
                            Ideal para ser exibido em um relatório executivo para stakeholders.
                            Baseie-se estritamente nesses dados para embasar suas análises.                        
                            Dados do Gráfico Plotly": {instance_df.to_dict(orient="records")}
                            "Valor previsto": {inverse_mapper[predicted[0]]},
                            "Valor Real": {inverse_mapper[y_selected]}                        
                            """)
                                
                st.markdown(response.content)    
    
    except Exception as error:
        st.error(f"Erro ao processar requisição. Tente novamente...\nErro: {error}")

elif graph_type=="Summary Plot" and generate:
    st.markdown("<h1 style='text-align: center'>Interpretabilidade de Modelo", unsafe_allow_html=True)
    st.markdown("<hr style=' border 1px; color: #50CCA0'>", unsafe_allow_html=True)
    try:
        with st.spinner("Gerando Interpretação do Gráfico com IA. Por favor aguarde um instante..."):
            graph_col, explain_col = st.columns([0.45, 0.55], gap="large")
            
            fig3, ax = plt.subplots()
            shap.summary_plot(shap_values[:500], X[:500], show=False)
            
            buffer = io.BytesIO()
            fig3.savefig(buffer, format="png")
            buffer.seek(0)        
        
            agno_fig = AgnoImage(content=buffer.read())

            with graph_col:
                st.subheader("Gráfico Summary - Plot")
                st.pyplot(fig3, width="stretch")
            
            with explain_col:
                st.subheader("Interpretação do Gráfico")
                if st.session_state["summary_plot"] is None:
                    response = agent.run(input="""Explique o Gráfico Summary - Plot de maneira simples eficiente para executivos.
                                        Foque nas Features mais importantes e preditivas para o modelo.
                                        O objetivo é Desmistificar os insights do Gráfico, possibilitando que qualquer pessoa
                                        possa entender o que o gráfico mostra através de linguagem de orientada à negócios.""",
                                        images=[agno_fig])
                    
                    st.session_state["summary_plot"] = response.content            
                    st.markdown(response.content)
                
                else:
                    st.markdown(st.session_state["summary_plot"])
            
    
    except Exception as error:
        st.error(f"Erro ao processar requisição. Tente novamente...\nErro: {error}")
        
        